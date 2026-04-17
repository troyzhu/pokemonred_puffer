# ADR 0001 — Project scope: goal-setting layer on top of `rubric_rl`

- **Date:** 2026-04-17
- **Status:** accepted

## Context

The baseline repo trains a Pokémon Red agent via PPO with ~30 hand-tuned
reward weights. A follow-up, `rubric_rl`, adds a GRPO trainer with a
static, meta-weighted `CompositeRubric`. Both work with a fixed reward
shape — they assume the designer already knows what "good play" looks
like.

The user wants to mimic how a **new player navigates an unknown game**:
set broad intent, discover milestones, revise the criteria-of-success
as they go. That requires a *self-revising* rubric rather than a static
one.

## Decision

Build a new `pokemonred_puffer/goal_rl/` module that wraps (not
replaces) `rubric_rl`. A new `train-goal` CLI command layers a goal
manager, hybrid eval triggers, a frozen evaluator, and an LLM revision
engine on top of the GRPO trainer.

## Consequences

- The static-rubric path (`train-grpo`) stays intact — no regression risk.
- Side-by-side baselining becomes possible (ADR 0015).
- Introduces new LLM dependency at run start (constitution parsing) and
  optionally mid-run (revision engine).
- New test surface under `tests/test_goal_rl.py`; existing tests untouched.

## Related

- [0002](0002-source-quality-tiers.md) source-quality policy
- [0003](0003-three-layer-schema.md) value schema
- [0016](0016-v1-implementation-landed.md) implementation status
