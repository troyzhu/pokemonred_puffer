"""Thin extension of CleanGRPO that plugs in the goal-setting loop.

Extends the existing GRPO trainer with:
1. Dynamic rubric — the trainer uses goal_manager.get_rubric() each epoch,
   so revisions mid-run reshape the training reward for the next epoch.
2. Eval cadence — frozen E1/E2/E3 evaluation on the hybrid trigger schedule.
3. Revision engine — LLM-driven, bounded, validated, with auto-rollback.
4. Audit trail — every revision, trigger, and hacking event is logged.

V1 simplification: the frozen eval runs on the snapshots collected during
training rather than a separate fixed-seed eval pass.  V2 can add a dedicated
eval loop.

Design reference: docs/goal_rl.md C05.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


from pokemonred_puffer.rubric_rl.grpo import CleanGRPO
from pokemonred_puffer.rubric_rl.rubrics import GameStateSnapshot

from pokemonred_puffer.goal_rl.evaluator_frozen import (
    EvalBatch,
    FrozenEvaluator,
    HackingEvent,
    HackingWatchdog,
)
from pokemonred_puffer.goal_rl.goal_manager import GoalManager, RevisionRecord
from pokemonred_puffer.goal_rl.revision_engine import (
    RevisionEngine,
    RevisionProposal,
)
from pokemonred_puffer.goal_rl.triggers import (
    MilestonePredicate,
    TriggerController,
    TriggerEvent,
)

logger = logging.getLogger(__name__)


# --- Config ---------------------------------------------------------------


@dataclass
class GoalRLRuntimeConfig:
    """Top-level runtime config for the goal-setting layer (alongside GRPOConfig)."""

    # Audit-trail file — if set, revisions/hacking events are appended as JSONL.
    audit_log_path: str | None = None
    # Soft ceiling on LLM revision calls per run.
    max_llm_revisions: int | None = None
    # Whether to dry-run (no LLM calls, always empty proposals).
    dry_run: bool = False


# --- Trainer --------------------------------------------------------------


@dataclass
class GoalGRPO(CleanGRPO):
    """GRPO with a dynamic, revision-aware rubric.

    Expects a pre-built GoalManager (which owns Layer-1/2/3 + custom criteria),
    TriggerController, FrozenEvaluator, HackingWatchdog, and RevisionEngine.

    The `rubric` field inherited from CleanGRPO is set to the goal_manager's
    current rubric at construction; run_epoch() refreshes it after any
    revision.
    """

    goal_manager: GoalManager = None  # type: ignore[assignment]
    trigger_controller: TriggerController = None  # type: ignore[assignment]
    frozen_evaluator: FrozenEvaluator = None  # type: ignore[assignment]
    hacking_watchdog: HackingWatchdog = None  # type: ignore[assignment]
    revision_engine: RevisionEngine = None  # type: ignore[assignment]
    goal_runtime_config: GoalRLRuntimeConfig = field(default_factory=GoalRLRuntimeConfig)

    # Trajectories captured across epochs.
    _training_reward_trajectory: list[float] = field(default_factory=list)
    _e2_trajectory: list[float] = field(default_factory=list)
    _milestone_predicates: list[MilestonePredicate] = field(default_factory=list)
    _prev_eval_snapshot: GameStateSnapshot | None = None
    _num_llm_revisions: int = 0

    def __post_init__(self):
        # Sanity-check required collaborators.
        for name in (
            "goal_manager",
            "trigger_controller",
            "frozen_evaluator",
            "hacking_watchdog",
            "revision_engine",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"GoalGRPO requires `{name}`")

        # Seed the parent's rubric with the goal manager's current composition.
        if self.rubric is None:
            self.rubric = self.goal_manager.get_rubric()

        super().__post_init__()

        # Ensure audit-log directory exists.
        if self.goal_runtime_config.audit_log_path:
            p = pathlib.Path(self.goal_runtime_config.audit_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.touch()

    # --- Override: run one epoch ------------------------------------------

    def run_epoch(self):
        # 1. Tick the trigger clock before anything else (so K counts this epoch).
        self.trigger_controller.tick()

        # 2. Refresh the training rubric from the goal manager — picks up the
        #    state after the previous epoch's revision (if any).
        self.rubric = self.goal_manager.get_rubric()
        if not self.rubric.rubrics:
            # Safety: if Layer-2 weights all went to zero (shouldn't happen),
            # fall back to the default rubric so training doesn't produce
            # zero reward.
            logger.warning("Goal manager yielded an empty rubric; using default.")
            from pokemonred_puffer.rubric_rl.rubrics import build_composite_rubric

            self.rubric = build_composite_rubric(None)

        # 3. Let the parent do the normal collect + evaluate_with_rubric + train.
        stats, infos = super().run_epoch()

        # 4. Gather snapshots that arrived this epoch.
        snapshots = [s for s in self._env_snapshots if s is not None]
        if not snapshots:
            # No completed episodes this epoch — nothing to evaluate.
            return stats, infos

        for snap in snapshots:
            self.goal_manager.record_snapshot(snap)
        latest = snapshots[-1]

        # 5. Track training reward for the revision engine's context.
        training_mean = float(self.losses.mean_reward)
        self._training_reward_trajectory.append(training_mean)

        # 6. Decide whether to run eval this epoch (milestone OR adaptive-K ceiling).
        ctx = self.goal_manager.get_primitive_context(latest)
        eval_event = self.trigger_controller.check_eval_trigger(ctx, self._milestone_predicates)
        if eval_event is None:
            return stats, infos

        # 7. Run frozen eval over the snapshots we have.
        eval_batch = self.frozen_evaluator.evaluate_batch(snapshots, epoch=self.epoch)
        self.trigger_controller.record_e2(eval_batch.e2_mean)
        self._e2_trajectory.append(eval_batch.e2_mean)
        # The latest snapshot becomes the "previous eval snapshot" for the
        # next round of delta / increment primitives.
        prev_eval_snapshot = self._prev_eval_snapshot
        self._prev_eval_snapshot = latest
        self.goal_manager.mark_eval_boundary(latest)

        self._log_eval(eval_batch, eval_event)

        # 8. Hacking watchdog on the (training_reward, E2) pair.
        hack_event = self.hacking_watchdog.record(
            training_reward=training_mean,
            e2=eval_batch.e2_mean,
            epoch=self.epoch,
        )
        if hack_event is not None:
            rolled_back = self.goal_manager.rollback_last_revision(reason=hack_event.detail)
            if rolled_back:
                self.hacking_watchdog.reset_after_rollback()
                self._log_hacking_rollback(hack_event)

        # 9. Revision trigger.
        rev_event = self.trigger_controller.check_revision_trigger(eval_event)
        if rev_event is None:
            return stats, infos

        # 10. Soft LLM-call ceiling.
        if (
            self.goal_runtime_config.max_llm_revisions is not None
            and self._num_llm_revisions >= self.goal_runtime_config.max_llm_revisions
        ):
            logger.info(
                "Skipping revision — LLM-call ceiling (%d) reached",
                self.goal_runtime_config.max_llm_revisions,
            )
            return stats, infos

        # 11. Invoke the revision engine (LLM or dry-run).
        proposal = self.revision_engine.propose(
            trigger=rev_event,
            manager=self.goal_manager,
            training_reward_trajectory=self._training_reward_trajectory,
            e2_trajectory=self._e2_trajectory,
            latest_snapshot=latest,
            prev_eval_snapshot=prev_eval_snapshot,
            milestone_predicates=self._milestone_predicates,
        )
        self._num_llm_revisions += 1

        if proposal.is_empty():
            logger.info(
                "Revision proposal empty (rationale=%s); skipping apply",
                (proposal.rationale or "")[:80],
            )
            self._log_revision_attempt(rev_event, proposal, applied=False)
            return stats, infos

        # 12. Apply revision.
        try:
            record = self.goal_manager.apply_revision(
                layer_two_deltas=proposal.layer_two_deltas,
                toggled_criteria=proposal.toggled_criteria,
                added_criteria=proposal.added_criteria,
                removed_criterion_names=proposal.removed_criterion_names,
                trigger=rev_event.kind,
                rationale=proposal.rationale,
                narrative=proposal.narrative,
                pre_e2=eval_batch.e2_mean,
            )
        except ValueError as e:
            logger.warning("Revision rejected at apply stage: %s", e)
            self._log_revision_attempt(rev_event, proposal, applied=False, error=str(e))
            return stats, infos

        # 13. Register any LLM-proposed milestone predicates.
        self._milestone_predicates.extend(proposal.milestone_predicates)

        self._log_revision_attempt(rev_event, proposal, applied=True, record=record)

        return stats, infos

    # --- Logging ----------------------------------------------------------

    def _write_audit(self, entry: dict[str, Any]) -> None:
        if not self.goal_runtime_config.audit_log_path:
            return
        try:
            with open(self.goal_runtime_config.audit_log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            logger.warning("Audit log write failed: %s", e)

    def _log_eval(self, batch: EvalBatch, event: TriggerEvent) -> None:
        entry = {
            "type": "eval",
            "ts": datetime.utcnow().isoformat(),
            "epoch": self.epoch,
            "trigger": event.to_dict(),
            "eval": batch.to_dict(),
        }
        self._write_audit(entry)
        if self.wandb_client is not None:
            log = {
                "goal_rl/e2_mean": batch.e2_mean,
                "goal_rl/e2_std": batch.e2_std,
                "goal_rl/eval_epoch": self.epoch,
                "goal_rl/adaptive_k": self.trigger_controller.adaptive_k(),
            }
            for fname, val in batch.dashboard_mean.items():
                log[f"goal_rl/dashboard/{fname}"] = val
            if batch.e3_mean is not None:
                log["goal_rl/e3_mean"] = batch.e3_mean
            self.wandb_client.log(log)

    def _log_hacking_rollback(self, event: HackingEvent) -> None:
        entry = {
            "type": "hacking_rollback",
            "ts": datetime.utcnow().isoformat(),
            "epoch": self.epoch,
            "event": event.to_dict(),
        }
        self._write_audit(entry)
        if self.wandb_client is not None:
            self.wandb_client.log(
                {
                    "goal_rl/hacking_events": len(self.hacking_watchdog.events),
                    "goal_rl/last_hack_training_delta": event.training_reward_delta,
                    "goal_rl/last_hack_e2_delta": event.e2_delta,
                }
            )

    def _log_revision_attempt(
        self,
        trigger: TriggerEvent,
        proposal: RevisionProposal,
        applied: bool,
        record: RevisionRecord | None = None,
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "type": "revision_attempt",
            "ts": datetime.utcnow().isoformat(),
            "epoch": self.epoch,
            "trigger": trigger.to_dict(),
            "applied": applied,
            "proposal": {
                "rationale": proposal.rationale,
                "narrative": proposal.narrative,
                "layer_two_deltas": proposal.layer_two_deltas,
                "added_criteria": [s.to_dict() for s in proposal.added_criteria],
                "removed_criteria": proposal.removed_criterion_names,
                "toggled_criteria": proposal.toggled_criteria,
                "milestone_predicates": [p.to_dict() for p in proposal.milestone_predicates],
                "warnings": proposal.warnings,
            },
        }
        if record is not None:
            entry["record"] = record.to_dict()
        if error:
            entry["error"] = error
        self._write_audit(entry)
        if self.wandb_client is not None:
            self.wandb_client.log(
                {
                    "goal_rl/num_revisions_applied": len(
                        [r for r in self.goal_manager.revision_log if r.trigger != "rollback"]
                    ),
                    "goal_rl/num_revision_attempts": self._num_llm_revisions,
                    "goal_rl/revision_applied": int(applied),
                }
            )


__all__ = [
    "GoalRLRuntimeConfig",
    "GoalGRPO",
]
