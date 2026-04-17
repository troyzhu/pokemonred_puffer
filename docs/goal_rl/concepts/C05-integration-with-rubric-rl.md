# C05 — Integration with `rubric_rl`

**Code:** `pokemonred_puffer/goal_rl/goal_grpo.py`,
`pokemonred_puffer/train.py::train_goal`
**Status:** adopted ([ADR 0001](../adr/0001-scope.md))

New module `pokemonred_puffer/goal_rl/` *wraps* and *calls*
`rubric_rl/` rather than replacing it. The existing static-rubric GRPO
path stays intact and callable; the new `train-goal` command layers
the goal manager on top.

## How the wiring works

- `GoalGRPO` subclasses `rubric_rl.grpo.CleanGRPO`.
- Each epoch, `self.rubric` is refreshed from
  `goal_manager.get_rubric()` before the inherited collect/train path
  runs — so revisions applied at the end of the previous epoch shape
  the next epoch's reward.
- Snapshots emitted by `RubricRewardEnv` in `info["rubric_snapshot"]`
  are captured by the parent class; `GoalGRPO` then feeds them to the
  `goal_manager`, `trigger_controller`, and `frozen_evaluator`.
- The frozen evaluator is constructed in a separate module
  ([C02](C02-frozen-evaluation-protocol.md)) with no import path to the
  revision engine — that's what "structurally unreachable" means in
  practice.

## Module layout

```
pokemonred_puffer/goal_rl/
├── __init__.py
├── primitives.py        # C03, C07 shared primitives
├── schema.py            # C01, C08 dataclasses + parser
├── goal_manager.py      # composes CompositeRubric; apply/rollback
├── triggers.py          # C04, C07 cadence + milestone detection
├── evaluator_frozen.py  # C02 frozen eval + watchdog
├── revision_engine.py   # C03 LLM proposer + validator
└── goal_grpo.py         # C05 integrated trainer
```

## Why wrap not extend

- The user already has a working, documented static-rubric system in
  `rubric_rl`. Keeping it intact is a regression-safety win.
- The goal layer is a legitimate superset of capabilities — wrapping
  expresses that cleanly.
- Allows side-by-side baselining in experiments
  ([ADR 0012](../adr/0012-experiment-design.md)).

## Related

- [C01](C01-three-layer-value-schema.md) schema
- [C02](C02-frozen-evaluation-protocol.md) eval
- [C03](C03-constrained-llm-revision-engine.md) revision engine
- [C04](C04-revision-triggers-and-safety-nets.md) triggers
