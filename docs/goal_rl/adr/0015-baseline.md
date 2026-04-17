# ADR 0015 — Baseline to beat: original repo PPO

- **Date:** 2026-04-17
- **Status:** accepted

## Context

The repo already contains a baseline PPO system with hand-tuned
reward weights (CARBS-tuned), documented in
`pokemonred_puffer/rewards/baseline.py` + the existing `train` command.
Per the README, it beats the game. That's the bar.

## Decision

The goal-setting system's first experimental target is to match or
beat this baseline on:

- **Primary metric — E2 mean across training** (ADR 0005).
- **Secondary — E1 dashboard distribution** (ADR 0005).
- **Tertiary — wall-clock-per-badge** and audit-trail inspection.

Experimental protocol is ADR 0012 (N=3 matched seeds, same compute
budget).

## Consequences

- The baseline has had significant tuning effort; goal-setting V1 has
  not. **Initial experiments should compare *trajectories* of
  improvement**, not just end scores — "does the goal-setting system
  *improve* faster or end higher per unit of tuning effort?" is a
  fairer question.
- E2 is defined to default to the *same* rubric that informs the
  baseline's reward weights — this is a feature, not a bug: "training
  rubric is hacking E2" becomes the same question as "training rubric
  is diverging from what the hand-tuned system learned."

## Related

- [0005](0005-frozen-evaluation.md) E2 defined as the canonical judge
- [0012](0012-experiment-design.md) experiment protocol
