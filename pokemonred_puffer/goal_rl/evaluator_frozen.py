"""Frozen three-layer evaluation protocol.

E1 — dashboard of raw facts (never weighted, never edited).
E2 — frozen scalar CompositeRubric (fixed at run start; canonical ruler).
E3 — optional session-target scalar rubric (frozen per run).

Evaluation is decoupled from revision: E1/E2/E3 run on the eval cadence
controlled by triggers.py.  The revision engine cannot touch this module;
the watchdog uses E2 history to detect reward-hacking and triggers rollbacks.

Design reference: docs/goal_rl.md C02.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from pokemonred_puffer.rubric_rl.rubrics import (
    CompositeRubric,
    GameStateSnapshot,
    build_composite_rubric,
)

logger = logging.getLogger(__name__)


# --- E1 dashboard ---------------------------------------------------------


# The raw-facts dashboard (C02).  Each entry is a snapshot field the designer
# wants to see a Pareto picture of, independent of any weighted scoring.
DASHBOARD_FIELDS: tuple[str, ...] = (
    "badges",
    "completed_required_events",
    "completed_required_items",
    "hm_count",
    "party_count",
    "max_level_sum",
    "seen_pokemon_count",
    "caught_pokemon_count",
    "unique_moves",
    "unique_maps_visited",
    "npcs_talked",
    "hidden_objs_found",
    "signs_read",
    "pokecenter_heals",
    "cut_tiles_used",
    "surf_tiles_used",
    "blackout_count",
    "total_steps",
)


def dashboard_from_snapshot(snapshot: GameStateSnapshot) -> dict[str, float]:
    """Extract the E1 dashboard vector from a snapshot."""
    out: dict[str, float] = {}
    for name in DASHBOARD_FIELDS:
        v = getattr(snapshot, name, None)
        if isinstance(v, (int, float)):
            out[name] = float(v)
    return out


# --- Eval result ----------------------------------------------------------


@dataclass
class EvalResult:
    """Result of evaluating a single snapshot with the frozen rubrics."""

    dashboard: dict[str, float]
    e2_scalar: float
    e3_scalar: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dashboard": dict(self.dashboard),
            "e2_scalar": self.e2_scalar,
            "e3_scalar": self.e3_scalar,
        }


@dataclass
class EvalBatch:
    """Aggregated eval over M episodes at a single cadence tick."""

    epoch: int
    per_episode: list[EvalResult]
    dashboard_mean: dict[str, float]
    dashboard_std: dict[str, float]
    e2_mean: float
    e2_std: float
    e3_mean: float | None = None
    e3_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "dashboard_mean": self.dashboard_mean,
            "dashboard_std": self.dashboard_std,
            "e2_mean": self.e2_mean,
            "e2_std": self.e2_std,
            "e3_mean": self.e3_mean,
            "e3_std": self.e3_std,
            "n_episodes": len(self.per_episode),
        }


# --- Frozen evaluator ------------------------------------------------------


class FrozenEvaluator:
    """Holds the E2 (and optionally E3) rubric; computes EvalResults.

    The E2 rubric is provided at construction and never modified.
    """

    def __init__(
        self,
        e2_rubric: CompositeRubric,
        e3_rubric: CompositeRubric | None = None,
    ):
        self.e2_rubric = e2_rubric
        self.e3_rubric = e3_rubric

    def evaluate_one(self, snapshot: GameStateSnapshot) -> EvalResult:
        dashboard = dashboard_from_snapshot(snapshot)
        e2 = self.e2_rubric.score(snapshot).total_score
        e3 = self.e3_rubric.score(snapshot).total_score if self.e3_rubric is not None else None
        return EvalResult(dashboard=dashboard, e2_scalar=float(e2), e3_scalar=e3)

    def evaluate_batch(self, snapshots: list[GameStateSnapshot], epoch: int) -> EvalBatch:
        per = [self.evaluate_one(s) for s in snapshots]
        return self._aggregate(per, epoch)

    @staticmethod
    def _aggregate(per: list[EvalResult], epoch: int) -> EvalBatch:
        if not per:
            return EvalBatch(
                epoch=epoch,
                per_episode=[],
                dashboard_mean={},
                dashboard_std={},
                e2_mean=0.0,
                e2_std=0.0,
                e3_mean=None,
                e3_std=None,
            )

        # Dashboard mean/std per field.
        fields = set()
        for r in per:
            fields.update(r.dashboard.keys())
        dash_mean: dict[str, float] = {}
        dash_std: dict[str, float] = {}
        for f in fields:
            vals = np.array([r.dashboard.get(f, 0.0) for r in per], dtype=np.float64)
            dash_mean[f] = float(vals.mean())
            dash_std[f] = float(vals.std())

        e2_vals = np.array([r.e2_scalar for r in per], dtype=np.float64)
        e3_has = all(r.e3_scalar is not None for r in per)
        e3_mean: float | None = None
        e3_std: float | None = None
        if e3_has:
            e3_vals = np.array([r.e3_scalar for r in per], dtype=np.float64)
            e3_mean = float(e3_vals.mean())
            e3_std = float(e3_vals.std())

        return EvalBatch(
            epoch=epoch,
            per_episode=per,
            dashboard_mean=dash_mean,
            dashboard_std=dash_std,
            e2_mean=float(e2_vals.mean()),
            e2_std=float(e2_vals.std()),
            e3_mean=e3_mean,
            e3_std=e3_std,
        )


# --- E2 rubric factory ----------------------------------------------------


def build_default_e2_rubric() -> CompositeRubric:
    """The canonical judge — reuses the baseline static rubric from rubric_rl.

    By default this uses `DEFAULT_RUBRIC_WEIGHTS` (story_progression=3.0,
    exploration=1.0, team_building=0.5, resource_management=0.3).  Anchoring
    E2 to this baseline means "training rubric is hacking E2" is the same
    question as "training rubric is diverging from the hand-tuned reward
    system we know works".
    """
    return build_composite_rubric(None)


def build_e2_rubric_from_config(config: Any) -> CompositeRubric:
    """Build E2 rubric from an optional config override.

    If config provides a `rubrics` key it's passed to the baseline factory;
    otherwise the default weights are used.
    """
    return build_composite_rubric(config)


# --- Hacking watchdog -----------------------------------------------------


@dataclass
class HackingEvent:
    """Record of a detected hack (training reward up, E2 not)."""

    epoch: int
    training_reward_delta: float
    e2_delta: float
    window: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "training_reward_delta": self.training_reward_delta,
            "e2_delta": self.e2_delta,
            "window": self.window,
            "detail": self.detail,
        }


class HackingWatchdog:
    """Detects likely reward-hacking of the training rubric.

    Heuristic: over a rolling window, if training-rubric mean reward rises by
    ≥ `training_rise_threshold` while E2 mean drops by ≥ `e2_drop_threshold`,
    flag it.  This is the C02 rollback trigger.
    """

    def __init__(
        self,
        window: int = 3,
        training_rise_threshold: float = 0.05,
        e2_drop_threshold: float = 0.05,
    ):
        self.window = max(1, window)
        self.training_rise_threshold = float(training_rise_threshold)
        self.e2_drop_threshold = float(e2_drop_threshold)
        self._training: list[float] = []
        self._e2: list[float] = []
        self.events: list[HackingEvent] = []

    def record(self, training_reward: float, e2: float, epoch: int = -1) -> HackingEvent | None:
        self._training.append(float(training_reward))
        self._e2.append(float(e2))

        if len(self._e2) < self.window + 1:
            return None

        t_recent = self._training[-(self.window + 1) :]
        e2_recent = self._e2[-(self.window + 1) :]
        t_delta = t_recent[-1] - t_recent[0]
        e2_delta = e2_recent[-1] - e2_recent[0]

        if t_delta >= self.training_rise_threshold and e2_delta <= -self.e2_drop_threshold:
            event = HackingEvent(
                epoch=epoch,
                training_reward_delta=t_delta,
                e2_delta=e2_delta,
                window=self.window,
                detail=(
                    f"Training reward +{t_delta:.3f} while E2 {e2_delta:.3f} "
                    f"over last {self.window} eval windows"
                ),
            )
            self.events.append(event)
            logger.warning("Hacking event detected: %s", event.detail)
            return event

        return None

    def reset_after_rollback(self) -> None:
        """After a rollback the watchdog's recent history is stale.

        Keep only the most recent sample as a seed (not necessary to keep,
        but prevents re-triggering on the same trailing window).
        """
        if self._training:
            self._training = [self._training[-1]]
            self._e2 = [self._e2[-1]]

    def summary(self) -> dict[str, Any]:
        return {
            "n_records": len(self._e2),
            "n_events": len(self.events),
            "latest_training": self._training[-1] if self._training else None,
            "latest_e2": self._e2[-1] if self._e2 else None,
        }


__all__ = [
    "DASHBOARD_FIELDS",
    "EvalResult",
    "EvalBatch",
    "FrozenEvaluator",
    "HackingEvent",
    "HackingWatchdog",
    "dashboard_from_snapshot",
    "build_default_e2_rubric",
    "build_e2_rubric_from_config",
]
