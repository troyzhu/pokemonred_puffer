"""Trigger control for eval and revision.

Three fire conditions for eval:
1. Any milestone detected this epoch (learned, see C07).
2. K-epoch safety-net elapsed since last eval (K is adaptive; see C04).

Three fire conditions for revision (subset of eval events):
1. Milestone-driven (almost always).
2. Plateau on E2 scalar across recent eval windows.
3. Revision ceiling: forced revision if none has fired in M epochs.

With cooldown W between revisions.  Design reference: docs/goal_rl.md C04 + C07.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


from pokemonred_puffer.goal_rl.primitives import (
    PrimitiveCall,
    PrimitiveContext,
    PrimitiveValidationError,
    build_evaluator,
    detect_milestones,
    validate_primitive_call,
)

logger = logging.getLogger(__name__)


# --- Config ---------------------------------------------------------------


@dataclass
class TriggerConfig:
    """Tunable thresholds for eval and revision triggers.

    Defaults match plan file "starting values": ε=0.02, N=3, K_min=2, K_max=10.
    The adaptive K uses a linear ramp over `total_epochs`.
    """

    # Adaptive K schedule.
    k_min: int = 2
    k_max: int = 10
    total_epochs: int = 100

    # Plateau detection (E2 scalar).
    plateau_epsilon: float = 0.02
    plateau_window: int = 3  # number of between-eval deltas to check

    # Revision gating.
    revision_ceiling_epochs: int = 30  # force revision if none for this long
    revision_cooldown_epochs: int = 3  # minimum gap between revisions


# --- Events ---------------------------------------------------------------


@dataclass
class TriggerEvent:
    """Describes why a trigger fired."""

    kind: str  # "milestone" | "safety_net" | "plateau" | "revision_ceiling"
    detail: str
    milestones: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "milestones": self.milestones,
        }


# --- Mutable state ---------------------------------------------------------


@dataclass
class TriggerState:
    epoch: int = 0
    last_eval_epoch: int = -1
    last_revision_epoch: int = -1
    e2_history: list[float] = field(default_factory=list)


# --- Milestone predicates (LLM-proposed) -----------------------------------


@dataclass
class MilestonePredicate:
    """A named predicate over a PrimitiveContext that signals a milestone.

    LLM-proposed; shares the primitives library with rubric criteria.
    """

    name: str
    description: str
    primitive_call: PrimitiveCall

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "primitive_call": self.primitive_call.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MilestonePredicate":
        return cls(
            name=str(d["name"]),
            description=str(d.get("description", "")),
            primitive_call=PrimitiveCall.from_dict(d["primitive_call"]),
        )


def validate_milestone_predicate(pred: MilestonePredicate) -> None:
    """Raise if the predicate is malformed."""
    validate_primitive_call(pred.primitive_call)


# --- Controller ------------------------------------------------------------


class TriggerController:
    """Decides when evals and revisions should fire.

    Usage (per training epoch):

        controller.tick()
        eval_event = controller.check_eval_trigger(context, predicates)
        if eval_event is not None:
            # run E1/E2/E3, get e2 score
            controller.record_e2(e2)
            rev_event = controller.check_revision_trigger(eval_event)
            if rev_event is not None:
                # invoke revision engine
                ...

    The controller is stateful: it tracks epoch counters and E2 history.
    """

    def __init__(self, config: TriggerConfig):
        self.config = config
        self.state = TriggerState()

    # --- Epoch tick ------------------------------------------------------

    def tick(self) -> None:
        self.state.epoch += 1

    # --- Adaptive K ------------------------------------------------------

    def adaptive_k(self) -> int:
        """Linear ramp K_min → K_max over total_epochs."""
        if self.config.total_epochs <= 0:
            return self.config.k_min
        t = min(max(self.state.epoch / max(self.config.total_epochs, 1), 0.0), 1.0)
        k = self.config.k_min + t * (self.config.k_max - self.config.k_min)
        return max(1, round(k))

    # --- Eval trigger -----------------------------------------------------

    def check_eval_trigger(
        self,
        context: PrimitiveContext,
        llm_predicates: list[MilestonePredicate] | None = None,
    ) -> TriggerEvent | None:
        """Decide if eval should run this epoch.

        Milestones fire eval immediately; otherwise a safety-net K-epoch
        ceiling applies.  Always runs at epoch 0 to seed E2 history.

        Returns None if no eval should run this epoch.
        """
        # Collect milestone candidates: auto-detected + LLM-predicate-driven.
        milestones = detect_milestones(context)
        if llm_predicates:
            for pred in llm_predicates:
                try:
                    evaluator = build_evaluator(pred.primitive_call)
                    if evaluator(context) > 0:
                        milestones.append(
                            {
                                "kind": "llm_predicate",
                                "name": pred.name,
                                "description": pred.description,
                            }
                        )
                except (PrimitiveValidationError, ValueError, TypeError) as e:
                    logger.warning("LLM milestone predicate %r failed: %s", pred.name, e)

        # Milestone → fire.
        if milestones:
            self.state.last_eval_epoch = self.state.epoch
            return TriggerEvent(
                kind="milestone",
                detail=f"{len(milestones)} milestone candidate(s)",
                milestones=milestones,
            )

        # Always run at epoch 0 to seed E2 history.
        if self.state.last_eval_epoch < 0:
            self.state.last_eval_epoch = self.state.epoch
            return TriggerEvent(kind="safety_net", detail="initial eval")

        # Safety-net K-epoch ceiling.
        k = self.adaptive_k()
        if self.state.epoch - self.state.last_eval_epoch >= k:
            self.state.last_eval_epoch = self.state.epoch
            return TriggerEvent(kind="safety_net", detail=f"{k} epochs elapsed (adaptive K)")

        return None

    # --- E2 history ------------------------------------------------------

    def record_e2(self, e2: float) -> None:
        """Called after an eval completes, with the E2 scalar."""
        self.state.e2_history.append(float(e2))

    # --- Revision trigger ------------------------------------------------

    def check_revision_trigger(self, last_eval: TriggerEvent) -> TriggerEvent | None:
        """Decide if a revision should fire given the latest eval event.

        Always called after an eval fires (last_eval is the eval's event).
        Returns None if no revision.
        """
        # Enforce cooldown.
        if self.state.last_revision_epoch >= 0:
            gap = self.state.epoch - self.state.last_revision_epoch
            if gap < self.config.revision_cooldown_epochs:
                return None

        # Milestone-driven → almost always revise.
        if last_eval.kind == "milestone":
            self.state.last_revision_epoch = self.state.epoch
            return TriggerEvent(
                kind="milestone",
                detail="milestone-driven revision",
                milestones=last_eval.milestones,
            )

        # Plateau.
        if self._is_plateau():
            self.state.last_revision_epoch = self.state.epoch
            return TriggerEvent(
                kind="plateau",
                detail=(
                    f"E2 flat over last {self.config.plateau_window} windows "
                    f"(ε={self.config.plateau_epsilon})"
                ),
            )

        # Revision ceiling.
        reference_epoch = (
            self.state.last_revision_epoch if self.state.last_revision_epoch >= 0 else 0
        )
        if self.state.epoch - reference_epoch >= self.config.revision_ceiling_epochs:
            self.state.last_revision_epoch = self.state.epoch
            return TriggerEvent(
                kind="revision_ceiling",
                detail=f"{self.config.revision_ceiling_epochs} epochs without revision",
            )

        return None

    # --- Plateau check ---------------------------------------------------

    def _is_plateau(self) -> bool:
        n = self.config.plateau_window
        history = self.state.e2_history
        if len(history) < n + 1:
            return False
        recent = history[-(n + 1) :]
        deltas = [recent[i + 1] - recent[i] for i in range(n)]
        # Plateau: no positive delta exceeds epsilon (allows regressions).
        return all(d < self.config.plateau_epsilon for d in deltas)

    # --- Observability ---------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "epoch": self.state.epoch,
            "last_eval_epoch": self.state.last_eval_epoch,
            "last_revision_epoch": self.state.last_revision_epoch,
            "adaptive_k": self.adaptive_k(),
            "e2_history_len": len(self.state.e2_history),
            "latest_e2": self.state.e2_history[-1] if self.state.e2_history else None,
        }


__all__ = [
    "TriggerConfig",
    "TriggerEvent",
    "TriggerState",
    "MilestonePredicate",
    "TriggerController",
    "validate_milestone_predicate",
]
