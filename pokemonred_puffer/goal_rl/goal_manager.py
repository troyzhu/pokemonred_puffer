"""Goal manager: owns Layer-1/2/3 state and composes the active CompositeRubric.

Tracks revision history, provides the PrimitiveContext for history-dependent
primitives, and applies revisions from the revision engine.

Built-in criteria per Layer-2 category are implemented as raw Python lambdas
(not PrimitiveCall-based).  LLM-proposed criteria use the primitives library
so the validator can check them.

Design reference: docs/goal_rl.md C01 + C03 + C05.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from pokemonred_puffer.rubric_rl.rubrics import (
    CompositeRubric,
    Criterion,
    GameStateSnapshot,
    Rubric,
)

from pokemonred_puffer.goal_rl.primitives import (
    PrimitiveCall,
    PrimitiveContext,
    build_evaluator,
    validate_primitive_call,
)
from pokemonred_puffer.goal_rl.schema import (
    GoalCategory,
    GoalRLConfig,
)

logger = logging.getLogger(__name__)


# --- Built-in criteria per category ---------------------------------------


def _progress_rubric() -> Rubric:
    return Rubric(
        name="progress",
        criteria=[
            Criterion(
                name="badges",
                description="Gym badges obtained (out of 8).",
                weight=10.0,
                evaluate=lambda s: s.badges / 8.0,
                category="progress",
            ),
            Criterion(
                name="required_events",
                description="Required story events cleared.",
                weight=7.0,
                evaluate=lambda s: s.completed_required_events / max(s.total_required_events, 1),
                category="progress",
            ),
            Criterion(
                name="required_items",
                description="Key items obtained.",
                weight=5.0,
                evaluate=lambda s: s.completed_required_items / max(s.total_required_items, 1),
                category="progress",
            ),
            Criterion(
                name="hm_count",
                description="HMs obtained (out of 5).",
                weight=8.0,
                evaluate=lambda s: s.hm_count / max(s.total_hms, 1),
                category="progress",
            ),
        ],
    )


def _completeness_rubric() -> Rubric:
    return Rubric(
        name="completeness",
        criteria=[
            Criterion(
                name="pokedex_caught",
                description="Pokedex entries caught.",
                weight=4.0,
                evaluate=lambda s: s.caught_pokemon_count / max(s.total_pokemon, 1),
                category="completeness",
            ),
            Criterion(
                name="pokedex_seen",
                description="Pokedex entries seen.",
                weight=2.0,
                evaluate=lambda s: s.seen_pokemon_count / max(s.total_pokemon, 1),
                category="completeness",
            ),
            Criterion(
                name="useful_items",
                description="Useful items collected.",
                weight=1.5,
                evaluate=lambda s: s.completed_useful_items / max(s.total_useful_items, 1),
                category="completeness",
            ),
        ],
    )


def _mastery_rubric() -> Rubric:
    return Rubric(
        name="mastery",
        criteria=[
            Criterion(
                name="team_level_sum",
                description="Sum of party levels (proxy for combat strength).",
                weight=4.0,
                evaluate=lambda s: min(s.max_level_sum / 300.0, 1.0),
                category="mastery",
            ),
            Criterion(
                name="team_size",
                description="Party size (more coverage).",
                weight=2.0,
                evaluate=lambda s: min(s.party_count / 6.0, 1.0),
                category="mastery",
            ),
            Criterion(
                name="unique_moves",
                description="Distinct moves learned across party.",
                weight=1.5,
                evaluate=lambda s: min(s.unique_moves / 20.0, 1.0),
                category="mastery",
            ),
        ],
    )


def _discovery_rubric() -> Rubric:
    return Rubric(
        name="discovery",
        criteria=[
            Criterion(
                name="maps_visited",
                description="Unique maps entered.",
                weight=3.0,
                evaluate=lambda s: min(s.unique_maps_visited / 100.0, 1.0),
                category="discovery",
            ),
            Criterion(
                name="tiles_explored",
                description="Unique tiles stepped on.",
                weight=2.0,
                evaluate=lambda s: min(s.exploration_tile_count / 5000.0, 1.0),
                category="discovery",
            ),
            Criterion(
                name="npcs_talked",
                description="NPCs interacted with.",
                weight=1.0,
                evaluate=lambda s: min(s.npcs_talked / 200.0, 1.0),
                category="discovery",
            ),
            Criterion(
                name="hidden_objects",
                description="Hidden objects discovered.",
                weight=1.5,
                evaluate=lambda s: min(s.hidden_objs_found / 50.0, 1.0),
                category="discovery",
            ),
        ],
    )


def _efficiency_rubric() -> Rubric:
    return Rubric(
        name="efficiency",
        criteria=[
            Criterion(
                name="progress_per_step",
                description="Required events cleared relative to step budget.",
                weight=3.0,
                evaluate=lambda s: (
                    (s.completed_required_events / max(s.total_required_events, 1))
                    * (1.0 - min(s.total_steps / max(s.max_steps, 1), 1.0))
                ),
                category="efficiency",
            ),
            Criterion(
                name="low_step_usage",
                description="Steps remaining in budget.",
                weight=1.5,
                evaluate=lambda s: 1.0 - min(s.total_steps / max(s.max_steps, 1), 1.0),
                category="efficiency",
            ),
        ],
    )


def _safety_rubric() -> Rubric:
    return Rubric(
        name="safety",
        criteria=[
            Criterion(
                name="no_blackout",
                description="Avoid blacking out (party faints).",
                weight=3.0,
                evaluate=lambda s: max(1.0 - s.blackout_count / 10.0, 0.0),
                category="safety",
            ),
            Criterion(
                name="pokecenter_usage",
                description="Uses pokecenter to recover.",
                weight=1.0,
                evaluate=lambda s: min(s.pokecenter_heals / 10.0, 1.0),
                category="safety",
            ),
        ],
    )


def _diversity_rubric() -> Rubric:
    return Rubric(
        name="diversity",
        criteria=[
            Criterion(
                name="move_diversity",
                description="Distinct moves across party.",
                weight=2.0,
                evaluate=lambda s: min(s.unique_moves / 20.0, 1.0),
                category="diversity",
            ),
            Criterion(
                name="pokedex_breadth",
                description="Breadth of species seen.",
                weight=1.5,
                evaluate=lambda s: min(s.seen_pokemon_count / 60.0, 1.0),
                category="diversity",
            ),
            Criterion(
                name="field_move_usage",
                description="Cut + Surf tiles exercised.",
                weight=1.0,
                evaluate=lambda s: min((s.cut_tiles_used + s.surf_tiles_used) / 100.0, 1.0),
                category="diversity",
            ),
        ],
    )


_BUILTIN_FACTORIES: dict[GoalCategory, Callable[[], Rubric]] = {
    GoalCategory.PROGRESS: _progress_rubric,
    GoalCategory.COMPLETENESS: _completeness_rubric,
    GoalCategory.MASTERY: _mastery_rubric,
    GoalCategory.DISCOVERY: _discovery_rubric,
    GoalCategory.EFFICIENCY: _efficiency_rubric,
    GoalCategory.SAFETY: _safety_rubric,
    GoalCategory.DIVERSITY: _diversity_rubric,
}


# --- LLM-proposed (primitive-based) criteria ------------------------------


@dataclass
class CriterionSpec:
    """A serializable spec for a criterion built from a PrimitiveCall.

    LLM revisions produce these; the validator checks them before we wire
    them into a Criterion that the existing rubric_rl pipeline understands.
    """

    name: str
    description: str
    weight: float
    primitive_call: PrimitiveCall
    category: GoalCategory
    # Whether this criterion came from an LLM revision (for audit).
    source: str = "llm_revision"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "primitive_call": self.primitive_call.to_dict(),
            "category": self.category.value,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CriterionSpec":
        return cls(
            name=str(d["name"]),
            description=str(d.get("description", "")),
            weight=float(d["weight"]),
            primitive_call=PrimitiveCall.from_dict(d["primitive_call"]),
            category=GoalCategory(d["category"]),
            source=str(d.get("source", "llm_revision")),
        )


def build_criterion_from_spec(
    spec: CriterionSpec, context_provider: Callable[[GameStateSnapshot], PrimitiveContext]
) -> Criterion:
    """Convert a validated CriterionSpec to a Criterion usable by rubric_rl.

    `context_provider(snapshot)` returns the up-to-date PrimitiveContext given
    the snapshot being evaluated.  The goal manager supplies this.
    """
    evaluator = build_evaluator(spec.primitive_call)

    def evaluate(snapshot: GameStateSnapshot) -> float:
        ctx = context_provider(snapshot)
        try:
            return evaluator(ctx)
        except (ValueError, TypeError) as e:
            logger.warning("Criterion %r failed to evaluate (%s); returning 0.0", spec.name, e)
            return 0.0

    return Criterion(
        name=spec.name,
        description=spec.description,
        weight=spec.weight,
        evaluate=evaluate,
        category=spec.category.value,
    )


# --- Revision records -----------------------------------------------------


@dataclass
class RevisionRecord:
    """Audit-trail entry for a single revision."""

    timestamp: str
    trigger: str  # "milestone", "plateau", "revision_ceiling", "manual", "rollback"
    rationale: str
    # Pre-revision snapshots of the editable state
    prev_layer_two: dict[str, float]
    prev_criterion_specs: list[dict[str, Any]]
    # What changed (as returned by the revision engine)
    layer_two_deltas: dict[str, float] = field(default_factory=dict)
    added_criteria: list[dict[str, Any]] = field(default_factory=list)
    removed_criterion_names: list[str] = field(default_factory=list)
    toggled_criteria: dict[str, bool] = field(default_factory=dict)
    # E2 scores around the revision, for watchdog
    pre_e2: float | None = None
    post_e2: float | None = None
    # LLM-proposed narrative note
    narrative: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "rationale": self.rationale,
            "prev_layer_two": self.prev_layer_two,
            "prev_criterion_specs": self.prev_criterion_specs,
            "layer_two_deltas": self.layer_two_deltas,
            "added_criteria": self.added_criteria,
            "removed_criterion_names": self.removed_criterion_names,
            "toggled_criteria": self.toggled_criteria,
            "pre_e2": self.pre_e2,
            "post_e2": self.post_e2,
            "narrative": self.narrative,
        }


# --- Goal manager ---------------------------------------------------------


# Bounds on Layer-2 category weight deltas per single revision (plan C03).
LAYER_TWO_DELTA_MIN = 0.75
LAYER_TWO_DELTA_MAX = 1.33
# Absolute bounds on sub-criterion weights (per plan).
SUB_WEIGHT_MIN = 0.0
SUB_WEIGHT_MAX = 10.0


class GoalManager:
    """Owns the active Layer-1/2/3 state and composes the current rubric.

    Responsibilities:
    - Compose a CompositeRubric from Layer-2 weights × (builtin + custom criteria).
    - Maintain a PrimitiveContext for history-dependent criteria.
    - Apply and roll back revisions proposed by the revision engine.
    - Maintain an audit trail of revisions for observability and debugging.

    Not responsible for:
    - Deciding when to revise (that's triggers.py).
    - Computing the frozen evaluator (that's evaluator_frozen.py).
    - Calling the LLM (that's revision_engine.py).
    """

    def __init__(self, config: GoalRLConfig):
        self.config = config
        self._history: list[GameStateSnapshot] = []
        self._last_eval_snapshot: GameStateSnapshot | None = None
        # Per-category list of LLM-proposed criterion specs (active).
        self._custom_specs: dict[GoalCategory, list[CriterionSpec]] = {
            cat: [] for cat in GoalCategory
        }
        # Toggle state for built-in criteria — name → enabled.
        # Structure: {category: {crit_name: bool}}.
        self._builtin_enabled: dict[GoalCategory, dict[str, bool]] = {}
        for cat, factory in _BUILTIN_FACTORIES.items():
            self._builtin_enabled[cat] = {c.name: True for c in factory().criteria}
        # Audit trail.
        self.revision_log: list[RevisionRecord] = []

    # --- Context for history-dependent primitives -------------------------

    def get_primitive_context(self, snapshot: GameStateSnapshot) -> PrimitiveContext:
        return PrimitiveContext(
            snapshot=snapshot,
            prev_snapshot=self._last_eval_snapshot,
            history=list(self._history),
        )

    def record_snapshot(self, snapshot: GameStateSnapshot) -> None:
        """Append a snapshot to the run history.  Called after each episode."""
        self._history.append(snapshot)

    def mark_eval_boundary(self, snapshot: GameStateSnapshot) -> None:
        """Set the last-eval snapshot used by delta/increment primitives."""
        self._last_eval_snapshot = snapshot

    # --- Rubric composition ------------------------------------------------

    def get_rubric(self) -> CompositeRubric:
        """Build the active CompositeRubric from current state."""
        rubrics: list[tuple[float, Rubric]] = []
        for cat in GoalCategory:
            meta_weight = self.config.layer_two.weights.get(cat, 0.0)
            if meta_weight <= 0:
                continue
            rubric = self._compose_category_rubric(cat)
            if rubric.criteria:
                rubrics.append((meta_weight, rubric))
        return CompositeRubric(rubrics=rubrics)

    def _compose_category_rubric(self, cat: GoalCategory) -> Rubric:
        """Collect enabled built-in + custom criteria for a single category."""
        factory = _BUILTIN_FACTORIES[cat]
        base = factory()
        enabled = self._builtin_enabled.get(cat, {})
        active_builtins = [c for c in base.criteria if enabled.get(c.name, True)]

        customs: list[Criterion] = []
        for spec in self._custom_specs[cat]:
            customs.append(build_criterion_from_spec(spec, self.get_primitive_context))

        return Rubric(name=cat.value, criteria=active_builtins + customs)

    def current_criterion_specs(self) -> list[dict[str, Any]]:
        """Snapshot of custom criterion specs across all categories (for audit)."""
        out: list[dict[str, Any]] = []
        for cat, specs in self._custom_specs.items():
            for spec in specs:
                out.append(spec.to_dict())
        return out

    # --- Revision application ---------------------------------------------

    def apply_revision(
        self,
        layer_two_deltas: dict[str, float] | None = None,
        toggled_criteria: dict[str, bool] | None = None,
        added_criteria: list[CriterionSpec] | None = None,
        removed_criterion_names: list[str] | None = None,
        trigger: str = "manual",
        rationale: str = "",
        narrative: str | None = None,
        pre_e2: float | None = None,
    ) -> RevisionRecord:
        """Apply a revision to the active state.

        Deltas on Layer-2 weights are bounded (LAYER_TWO_DELTA_MIN, MAX).
        Returns the RevisionRecord describing what happened.

        Raises ValueError if bounds or schema checks fail.
        """
        prev_layer_two = dict(self.config.layer_two.to_dict())
        prev_specs = self.current_criterion_specs()

        # --- Layer 2 weight deltas (multiplicative, bounded) ---
        applied_deltas: dict[str, float] = {}
        if layer_two_deltas:
            for cat_name, mult in layer_two_deltas.items():
                try:
                    cat = GoalCategory(cat_name)
                except ValueError:
                    logger.warning("Ignoring unknown Layer-2 category %r", cat_name)
                    continue
                mult = float(mult)
                if not (LAYER_TWO_DELTA_MIN <= mult <= LAYER_TWO_DELTA_MAX):
                    raise ValueError(
                        f"Layer-2 multiplicative delta for {cat_name}={mult} outside "
                        f"[{LAYER_TWO_DELTA_MIN}, {LAYER_TWO_DELTA_MAX}]"
                    )
                old = self.config.layer_two.weights.get(cat, 0.0)
                new = max(0.0, min(old * mult, SUB_WEIGHT_MAX))
                self.config.layer_two.weights[cat] = new
                applied_deltas[cat_name] = mult

        # --- Built-in criterion toggles ---
        applied_toggles: dict[str, bool] = {}
        if toggled_criteria:
            for crit_name, enabled in toggled_criteria.items():
                # crit_name must be "category:crit" e.g. "progress:badges"
                if ":" not in crit_name:
                    logger.warning(
                        "Toggled criterion %r missing category prefix; expected 'category:name'",
                        crit_name,
                    )
                    continue
                cat_name, name = crit_name.split(":", 1)
                try:
                    cat = GoalCategory(cat_name)
                except ValueError:
                    logger.warning("Unknown toggle category %r", cat_name)
                    continue
                if name not in self._builtin_enabled.get(cat, {}):
                    logger.warning("Unknown toggle criterion %r in category %s", name, cat_name)
                    continue
                self._builtin_enabled[cat][name] = bool(enabled)
                applied_toggles[crit_name] = bool(enabled)

        # --- Add LLM-proposed criteria ---
        applied_added: list[dict[str, Any]] = []
        if added_criteria:
            for spec in added_criteria:
                # Validator must have already accepted this; re-check defensively.
                validate_primitive_call(spec.primitive_call)
                if not (SUB_WEIGHT_MIN <= spec.weight <= SUB_WEIGHT_MAX):
                    raise ValueError(
                        f"Criterion {spec.name} weight {spec.weight} outside "
                        f"[{SUB_WEIGHT_MIN}, {SUB_WEIGHT_MAX}]"
                    )
                # Name collision check: disallow shadowing existing names in the category.
                existing = {c.name for c in _BUILTIN_FACTORIES[spec.category]().criteria} | {
                    c.name for c in self._custom_specs[spec.category]
                }
                if spec.name in existing:
                    raise ValueError(
                        f"Criterion name {spec.name!r} already exists in category {spec.category.value}"
                    )
                self._custom_specs[spec.category].append(spec)
                applied_added.append(spec.to_dict())

        # --- Remove LLM-proposed criteria ---
        applied_removed: list[str] = []
        if removed_criterion_names:
            for name in removed_criterion_names:
                for cat in GoalCategory:
                    before = len(self._custom_specs[cat])
                    self._custom_specs[cat] = [s for s in self._custom_specs[cat] if s.name != name]
                    if len(self._custom_specs[cat]) < before:
                        applied_removed.append(name)

        record = RevisionRecord(
            timestamp=datetime.utcnow().isoformat(),
            trigger=trigger,
            rationale=rationale,
            prev_layer_two=prev_layer_two,
            prev_criterion_specs=prev_specs,
            layer_two_deltas=applied_deltas,
            toggled_criteria=applied_toggles,
            added_criteria=applied_added,
            removed_criterion_names=applied_removed,
            narrative=narrative,
            pre_e2=pre_e2,
        )
        self.revision_log.append(record)
        logger.info(
            "Applied revision (trigger=%s): L2 deltas=%s, +%d criteria, -%d criteria, %d toggles",
            trigger,
            applied_deltas,
            len(applied_added),
            len(applied_removed),
            len(applied_toggles),
        )
        return record

    def rollback_last_revision(self, reason: str = "") -> bool:
        """Undo the most recent revision.  Returns True if one was rolled back.

        The rollback itself is logged as a new revision record (trigger='rollback')
        so the history remains append-only.
        """
        if not self.revision_log:
            return False
        last = self.revision_log[-1]

        # Restore Layer-2 weights
        restored_l2: dict[GoalCategory, float] = {}
        for k, v in last.prev_layer_two.items():
            try:
                restored_l2[GoalCategory(k)] = float(v)
            except ValueError:
                continue
        self.config.layer_two.weights = restored_l2
        self.config.layer_two.__post_init__()  # re-canonicalize

        # Restore custom criteria
        new_specs: dict[GoalCategory, list[CriterionSpec]] = {cat: [] for cat in GoalCategory}
        for d in last.prev_criterion_specs:
            try:
                spec = CriterionSpec.from_dict(d)
                new_specs[spec.category].append(spec)
            except (ValueError, KeyError) as e:
                logger.warning("Could not restore criterion spec: %s", e)
        self._custom_specs = new_specs

        # We do not try to restore toggle state from before the previous revision —
        # that would require tracking toggles-before-last instead of the pre-revision
        # snapshot.  V1 accepts that rollback restores weights and criteria but not
        # toggle state.  If this matters we can extend prev_builtin_enabled too.

        self.revision_log.append(
            RevisionRecord(
                timestamp=datetime.utcnow().isoformat(),
                trigger="rollback",
                rationale=reason,
                prev_layer_two=dict(last.prev_layer_two),
                prev_criterion_specs=copy.deepcopy(last.prev_criterion_specs),
            )
        )
        logger.warning("Rolled back last revision (reason=%s)", reason)
        return True

    # --- Observability ----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a compact snapshot of the current state for logging."""
        return {
            "layer_two_weights": self.config.layer_two.to_dict(),
            "num_custom_criteria": sum(len(v) for v in self._custom_specs.values()),
            "num_revisions": len(self.revision_log),
            "constraints": self.config.layer_three.to_list(),
        }


__all__ = [
    "CriterionSpec",
    "RevisionRecord",
    "GoalManager",
    "build_criterion_from_spec",
    "LAYER_TWO_DELTA_MIN",
    "LAYER_TWO_DELTA_MAX",
    "SUB_WEIGHT_MIN",
    "SUB_WEIGHT_MAX",
]
