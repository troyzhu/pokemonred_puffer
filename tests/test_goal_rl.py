"""Unit tests for the goal_rl module.

Covers:
- primitives: evaluation, validation, field checks, auto-milestone detection
- schema: heuristic constitution parsing, Layer-2 canonicalization
- goal_manager: apply + rollback revisions, custom criteria composition
- revision_engine: parse valid / invalid LLM responses
- triggers: adaptive K, plateau detection, cooldown, eval vs revision triggers
- evaluator_frozen: dashboard extraction, hacking watchdog
"""

from __future__ import annotations

import pytest

from pokemonred_puffer.rubric_rl.rubrics import GameStateSnapshot

from pokemonred_puffer.goal_rl import primitives
from pokemonred_puffer.goal_rl.evaluator_frozen import (
    FrozenEvaluator,
    HackingWatchdog,
    build_default_e2_rubric,
    dashboard_from_snapshot,
)
from pokemonred_puffer.goal_rl.goal_manager import (
    CriterionSpec,
    GoalManager,
)
from pokemonred_puffer.goal_rl.primitives import (
    PrimitiveCall,
    PrimitiveContext,
    PrimitiveValidationError,
    detect_milestones,
    validate_primitive_call,
)
from pokemonred_puffer.goal_rl.revision_engine import (
    RevisionValidationError,
    parse_revision,
)
from pokemonred_puffer.goal_rl.schema import (
    GoalCategory,
    LayerTwo,
    build_config,
)
from pokemonred_puffer.goal_rl.triggers import (
    TriggerConfig,
    TriggerController,
)


# --- primitives ------------------------------------------------------------


class TestPrimitives:
    def test_fraction_clips(self):
        ctx = PrimitiveContext(snapshot=GameStateSnapshot(badges=4))
        assert primitives.fraction(ctx, field="badges", max=8) == 0.5
        # Above max should clip to 1.0
        assert primitives.fraction(ctx, field="badges", max=2) == 1.0
        # Non-positive max → 0
        assert primitives.fraction(ctx, field="badges", max=0) == 0.0

    def test_inverse_fraction(self):
        ctx = PrimitiveContext(snapshot=GameStateSnapshot(blackout_count=3))
        assert primitives.inverse_fraction(ctx, field="blackout_count", max=10) == pytest.approx(
            0.7
        )
        # Zero field → full credit
        ctx_zero = PrimitiveContext(snapshot=GameStateSnapshot(blackout_count=0))
        assert primitives.inverse_fraction(ctx_zero, field="blackout_count", max=10) == 1.0

    def test_delta_requires_prev(self):
        cur = GameStateSnapshot(badges=5)
        prev = GameStateSnapshot(badges=2)
        # No prev → 0
        assert primitives.delta(PrimitiveContext(snapshot=cur), field="badges") == 0.0
        # With prev → difference
        ctx = PrimitiveContext(snapshot=cur, prev_snapshot=prev)
        assert primitives.delta(ctx, field="badges") == 3.0

    def test_first_time(self):
        cur = GameStateSnapshot(badges=1)
        # No history, cur > 0 → first time
        ctx = PrimitiveContext(snapshot=cur, history=[])
        assert primitives.first_time(ctx, field="badges") == 1.0
        # History contains earlier >0 → not first time
        ctx = PrimitiveContext(snapshot=cur, history=[GameStateSnapshot(badges=1)])
        assert primitives.first_time(ctx, field="badges") == 0.0
        # Current zero → not first time
        ctx = PrimitiveContext(snapshot=GameStateSnapshot(badges=0), history=[])
        assert primitives.first_time(ctx, field="badges") == 0.0

    def test_threshold_hit_and_cross(self):
        cur = GameStateSnapshot(badges=3)
        prev = GameStateSnapshot(badges=2)
        ctx = PrimitiveContext(snapshot=cur, prev_snapshot=prev)
        # 3 >= 3
        assert primitives.threshold_hit(ctx, field="badges", thresh=3) == 1.0
        assert primitives.threshold_hit(ctx, field="badges", thresh=4) == 0.0
        # 2 < 3 <= 3 → crossed upward
        assert primitives.threshold_cross(ctx, field="badges", thresh=3) == 1.0
        # Not a new crossing if prev already >= thresh
        ctx2 = PrimitiveContext(snapshot=cur, prev_snapshot=GameStateSnapshot(badges=3))
        assert primitives.threshold_cross(ctx2, field="badges", thresh=3) == 0.0

    def test_ratio_with_zero_denominator(self):
        ctx = PrimitiveContext(snapshot=GameStateSnapshot(badges=4, party_count=0))
        val = primitives.ratio(ctx, field_a="badges", field_b="party_count")
        # Should not blow up; divided by epsilon so value is large finite
        assert val > 1e6

    def test_rolling_avg(self):
        cur = GameStateSnapshot(max_level_sum=50)
        hist = [
            GameStateSnapshot(max_level_sum=10),
            GameStateSnapshot(max_level_sum=20),
            GameStateSnapshot(max_level_sum=30),
        ]
        ctx = PrimitiveContext(snapshot=cur, history=hist)
        # Window 3: [hist[-2], hist[-1], cur] = [20, 30, 50] → 33.33...
        assert primitives.rolling_avg(ctx, field="max_level_sum", window=3) == pytest.approx(
            (20 + 30 + 50) / 3
        )

    def test_validator_rejects_unknown_primitive(self):
        call = PrimitiveCall(primitive="no_such", args={"field": "badges"})
        with pytest.raises(PrimitiveValidationError):
            validate_primitive_call(call)

    def test_validator_rejects_unknown_field(self):
        call = PrimitiveCall(primitive="fraction", args={"field": "made_up_field", "max": 1.0})
        with pytest.raises(PrimitiveValidationError):
            validate_primitive_call(call)

    def test_validator_requires_args(self):
        call = PrimitiveCall(primitive="fraction", args={"field": "badges"})
        with pytest.raises(PrimitiveValidationError):
            validate_primitive_call(call)

    def test_auto_milestone_detection(self):
        prev = GameStateSnapshot(badges=0, hm_count=0)
        cur = GameStateSnapshot(badges=1, hm_count=1)
        ctx = PrimitiveContext(snapshot=cur, prev_snapshot=prev, history=[prev])
        milestones = detect_milestones(ctx)
        kinds = {(m["kind"], m["field"]) for m in milestones}
        # Both are first-time since history has only a snapshot where both were 0.
        assert ("first_time", "badges") in kinds
        assert ("first_time", "hm_count") in kinds

    def test_auto_milestone_no_change(self):
        prev = GameStateSnapshot(badges=2)
        cur = GameStateSnapshot(badges=2)
        ctx = PrimitiveContext(
            snapshot=cur, prev_snapshot=prev, history=[GameStateSnapshot(badges=2)]
        )
        milestones = detect_milestones(ctx)
        assert all(m["field"] != "badges" for m in milestones)


# --- schema ---------------------------------------------------------------


class TestSchema:
    def test_heuristic_nuzlocke(self):
        cfg = build_config("Doing a Nuzlocke run aiming for 5 badges, safety first.")
        assert cfg.constitution.playstyle == "nuzlocker"
        assert cfg.constitution.target_badges == 5
        kinds = [c.kind.value for c in cfg.layer_three.constraints]
        assert "nuzlocke" in kinds
        # Nuzlocker defaults should have safety weight set.
        assert cfg.layer_two.weights[GoalCategory.SAFETY] > 0

    def test_heuristic_monotype(self):
        cfg = build_config("Monotype electric team, focus on pokedex.")
        kinds = [c.kind.value for c in cfg.layer_three.constraints]
        assert "monotype" in kinds
        # The "pokedex" keyword triggers pokedex_hunter playstyle.
        assert cfg.layer_two.weights[GoalCategory.COMPLETENESS] == 1.0

    def test_empty_constitution(self):
        cfg = build_config("")
        # Equal weights across all categories.
        weights = cfg.layer_two.to_dict()
        assert all(v == 1.0 for v in weights.values())
        assert not cfg.layer_three.constraints

    def test_layer_two_canonicalizes_unknown_category(self):
        lt = LayerTwo.from_dict({"progress": 2.0, "not_a_category": 99.0})
        assert lt.weights[GoalCategory.PROGRESS] == 2.0
        # Unknown category silently dropped.
        assert "not_a_category" not in {k.value for k in lt.weights}

    def test_layer_two_clips_weights(self):
        lt = LayerTwo.from_dict({"progress": -5.0, "mastery": 50.0})
        assert lt.weights[GoalCategory.PROGRESS] == 0.0
        assert lt.weights[GoalCategory.MASTERY] == 10.0  # clipped at 10


# --- goal_manager ---------------------------------------------------------


class TestGoalManager:
    def _make_manager(self, text: str = ""):
        cfg = build_config(text)
        return cfg, GoalManager(cfg)

    def test_rubric_composition(self):
        cfg, mgr = self._make_manager("Story runner aiming for 4 badges.")
        rubric = mgr.get_rubric()
        # Should have at least the `progress` rubric.
        names = {r.name for _, r in rubric.rubrics}
        assert "progress" in names

    def test_apply_revision_adjusts_weights(self):
        cfg, mgr = self._make_manager("")
        # Start with equal weights.
        before = cfg.layer_two.weights[GoalCategory.PROGRESS]
        mgr.apply_revision(layer_two_deltas={"progress": 1.25}, trigger="test")
        after = cfg.layer_two.weights[GoalCategory.PROGRESS]
        assert after == pytest.approx(before * 1.25)

    def test_apply_revision_rejects_out_of_bounds(self):
        _, mgr = self._make_manager("")
        with pytest.raises(ValueError):
            mgr.apply_revision(layer_two_deltas={"progress": 5.0})

    def test_add_and_remove_custom_criterion(self):
        _, mgr = self._make_manager("")
        spec = CriterionSpec(
            name="crossed_three_badges",
            description="Hit 3 badges.",
            weight=2.0,
            primitive_call=PrimitiveCall(
                primitive="threshold_hit",
                args={"field": "badges", "thresh": 3},
            ),
            category=GoalCategory.PROGRESS,
        )
        mgr.apply_revision(added_criteria=[spec], trigger="milestone")
        rubric = mgr.get_rubric()
        # The progress rubric should include our new criterion.
        progress = next(r for _, r in rubric.rubrics if r.name == "progress")
        assert any(c.name == "crossed_three_badges" for c in progress.criteria)

        # Remove it.
        mgr.apply_revision(removed_criterion_names=["crossed_three_badges"], trigger="test")
        rubric2 = mgr.get_rubric()
        progress2 = next(r for _, r in rubric2.rubrics if r.name == "progress")
        assert all(c.name != "crossed_three_badges" for c in progress2.criteria)

    def test_rollback_restores_weights(self):
        cfg, mgr = self._make_manager("")
        before = dict(cfg.layer_two.to_dict())
        mgr.apply_revision(layer_two_deltas={"progress": 1.25}, trigger="test")
        assert cfg.layer_two.to_dict() != before
        ok = mgr.rollback_last_revision(reason="unit test")
        assert ok
        assert cfg.layer_two.to_dict() == before
        # Rollback itself is logged.
        assert len(mgr.revision_log) == 2
        assert mgr.revision_log[-1].trigger == "rollback"


# --- revision_engine ------------------------------------------------------


class TestRevisionParser:
    def test_parse_clean(self):
        raw = """
        {
          "rationale": "testing",
          "narrative": "note",
          "layer_two_deltas": {"progress": 1.2},
          "added_criteria": [],
          "removed_criteria": [],
          "toggled_criteria": {},
          "milestone_predicates": []
        }
        """
        prop = parse_revision(raw)
        assert prop.layer_two_deltas == {"progress": 1.2}
        assert prop.is_empty() is False  # has a delta

    def test_parse_fenced(self):
        raw = """```json
        {"rationale": "x", "layer_two_deltas": {"mastery": 0.9}}
        ```"""
        prop = parse_revision(raw)
        assert prop.layer_two_deltas == {"mastery": 0.9}

    def test_parse_clips_out_of_bounds_delta(self):
        raw = """{"rationale": "x", "layer_two_deltas": {"progress": 5.0}}"""
        prop = parse_revision(raw)
        # 5.0 > 1.33 → clipped to 1.33
        assert prop.layer_two_deltas["progress"] == 1.33
        assert any("clipping" in w for w in prop.warnings)

    def test_parse_filters_unknown_category(self):
        raw = """{"rationale": "x", "layer_two_deltas": {"progress": 1.0, "junk": 1.0}}"""
        prop = parse_revision(raw)
        assert "junk" not in prop.layer_two_deltas
        assert "progress" in prop.layer_two_deltas

    def test_parse_filters_bad_criterion(self):
        raw = """{
          "rationale": "x",
          "added_criteria": [
            {"name": "ok", "weight": 1.0, "category": "progress",
             "primitive_call": {"primitive": "fraction", "args": {"field": "badges", "max": 8}}},
            {"name": "bad_prim", "weight": 1.0, "category": "progress",
             "primitive_call": {"primitive": "no_such", "args": {}}}
          ]
        }"""
        prop = parse_revision(raw)
        assert {s.name for s in prop.added_criteria} == {"ok"}
        assert any("bad_prim" in w or "no_such" in w for w in prop.warnings)

    def test_parse_unparseable_raises(self):
        with pytest.raises(RevisionValidationError):
            parse_revision("totally not json at all")


# --- triggers -------------------------------------------------------------


class TestTriggers:
    def test_adaptive_k_ramp(self):
        cfg = TriggerConfig(k_min=2, k_max=10, total_epochs=100)
        tc = TriggerController(cfg)
        tc.state.epoch = 0
        assert tc.adaptive_k() == 2
        tc.state.epoch = 50
        assert tc.adaptive_k() == pytest.approx(6, abs=1)
        tc.state.epoch = 100
        assert tc.adaptive_k() == 10
        # Beyond total_epochs: clamped.
        tc.state.epoch = 200
        assert tc.adaptive_k() == 10

    def test_eval_triggers_on_milestone(self):
        cfg = TriggerConfig(k_min=100, k_max=100, total_epochs=100)
        tc = TriggerController(cfg)
        tc.tick()
        # Snapshot with badges=1, no prior history → first_time fires.
        ctx = PrimitiveContext(snapshot=GameStateSnapshot(badges=1))
        ev = tc.check_eval_trigger(ctx)
        assert ev is not None
        assert ev.kind == "milestone"

    def test_eval_safety_net_fires_without_milestone(self):
        cfg = TriggerConfig(k_min=2, k_max=2, total_epochs=10)
        tc = TriggerController(cfg)
        # Epoch 1: initial eval fires (seed).
        tc.tick()
        no_milestone_ctx = PrimitiveContext(
            snapshot=GameStateSnapshot(badges=2),
            prev_snapshot=GameStateSnapshot(badges=2),
            history=[GameStateSnapshot(badges=2)],
        )
        ev = tc.check_eval_trigger(no_milestone_ctx)
        assert ev is not None
        assert ev.kind == "safety_net"
        # Epoch 2: gap=1 < K=2 → no eval.
        tc.tick()
        ev = tc.check_eval_trigger(no_milestone_ctx)
        assert ev is None
        # Epoch 3: gap=2 ≥ K=2 → eval fires.
        tc.tick()
        ev = tc.check_eval_trigger(no_milestone_ctx)
        assert ev is not None
        assert ev.kind == "safety_net"

    def test_plateau_detection(self):
        cfg = TriggerConfig(
            k_min=1, k_max=1, total_epochs=10, plateau_epsilon=0.01, plateau_window=3
        )
        tc = TriggerController(cfg)
        # Not enough history yet.
        assert not tc._is_plateau()
        for v in [0.1, 0.15, 0.20, 0.25]:
            tc.record_e2(v)
        # Still rising by 0.05 each → not a plateau.
        assert not tc._is_plateau()
        for v in [0.25, 0.251, 0.2505, 0.251]:
            tc.record_e2(v)
        # Flat-ish → plateau.
        assert tc._is_plateau()

    def test_revision_cooldown(self):
        cfg = TriggerConfig(
            k_min=1,
            k_max=1,
            total_epochs=10,
            revision_cooldown_epochs=3,
            plateau_window=3,
            plateau_epsilon=0.01,
        )
        tc = TriggerController(cfg)
        tc.tick()  # epoch 1
        # Milestone-driven eval + revision at epoch 1.
        from pokemonred_puffer.goal_rl.triggers import TriggerEvent

        fake_eval = TriggerEvent(kind="milestone", detail="test", milestones=[{"k": "v"}])
        rev = tc.check_revision_trigger(fake_eval)
        assert rev is not None
        # Within cooldown: no revision.
        tc.tick()  # epoch 2 — gap=1 < cooldown=3
        rev2 = tc.check_revision_trigger(fake_eval)
        assert rev2 is None


# --- evaluator_frozen -----------------------------------------------------


class TestFrozenEvaluator:
    def test_dashboard_extraction(self):
        snap = GameStateSnapshot(badges=3, hm_count=1, party_count=4)
        dash = dashboard_from_snapshot(snap)
        assert dash["badges"] == 3.0
        assert dash["hm_count"] == 1.0
        assert dash["party_count"] == 4.0

    def test_e2_rubric_and_batch(self):
        e2 = build_default_e2_rubric()
        ev = FrozenEvaluator(e2_rubric=e2)
        snaps = [GameStateSnapshot(badges=i) for i in range(1, 4)]
        batch = ev.evaluate_batch(snaps, epoch=1)
        assert batch.e2_mean >= 0.0
        assert batch.e2_mean <= 1.0
        assert batch.dashboard_mean["badges"] == 2.0

    def test_watchdog_normal_training(self):
        wd = HackingWatchdog(window=2, training_rise_threshold=0.05, e2_drop_threshold=0.05)
        # Normal: both rise together.
        for t, e in [(0.1, 0.1), (0.2, 0.15), (0.3, 0.2)]:
            wd.record(t, e)
        assert not wd.events

    def test_watchdog_detects_hack(self):
        wd = HackingWatchdog(window=2, training_rise_threshold=0.05, e2_drop_threshold=0.05)
        # Hack pattern: training shoots up, E2 falls.
        for t, e in [(0.1, 0.3), (0.3, 0.25), (0.6, 0.10)]:
            wd.record(t, e)
        assert len(wd.events) == 1
        ev = wd.events[0]
        assert ev.training_reward_delta > 0
        assert ev.e2_delta < 0
