# ADR 0012 — First experiment: N=3 matched seeds vs baseline PPO

- **Date:** 2026-04-17
- **Status:** accepted

## Context

Once the implementation lands we need a protocol for "does this
actually help?" Pokémon Red RL is known to have high cross-seed
variance — a single seed comparison can show the baseline winning or
losing by luck alone. The user questioned whether seeds matter; they
do, because of the environment's stochasticity (encounters, RNG).

## Decision

First real experiment:

- **N = 3 seeds per system.** Larger N is better; 3 is the floor for
  reportable mean ± std.
- **Matched seeds** — the same `(env_seed, policy_seed)` pair for row
  i of baseline and row i of goal-setting. RNG-driven variance cancels
  between paired rows.
- **Identical compute budget** per system (wall-clock or env steps,
  whichever is tighter).
- **Primary metric:** E2 mean trajectory across training (ADR 0005).
  The frozen scalar is the ruler that doesn't move.
- **Secondary metric:** E1 dashboard distribution. Pareto picture of
  what's improving (badges, pokédex, maps, blackouts, …).
- **Tertiary:** wall-clock-per-badge; qualitative audit-trail
  inspection (do the LLM-proposed revisions look sensible?).

Compare *trajectories* of improvement, not just end-of-run scores —
the baseline has had significant tuning effort that the goal-setting
V1 has not.

## Consequences

- 6 training runs for the first experiment (N=3 × 2 systems).
- Need to ensure the heuristic constitution parser (ADR 0009) is used
  when running without LLM credentials, so matched seeds produce truly
  deterministic configs.
- The audit trail itself becomes a research artefact — reviewing LLM
  revisions is part of the evaluation.

## Related

- [0005](0005-frozen-evaluation.md) metrics
- [0015](0015-baseline.md) baseline definition
