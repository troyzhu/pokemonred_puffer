# ADR 0008 — Hybrid eval cadence with adaptive K safety net

- **Date:** 2026-04-17
- **Status:** accepted

## Context

Evaluating at a fixed K-epoch cadence is simple but either wasteful
(eval too often early when learning is fast) or unresponsive (eval too
rarely early when the game's phase is shifting). Milestone-only eval
is the opposite — great when events fire, but nothing when the agent
gets stuck.

User framing: the safety-net interval is the **worst case** for eval
spacing, not the target; milestones should beat it in the typical case.
User also suggested denser evals early in training when the agent is
still learning the basics.

## Decision

**Hybrid cadence:** eval fires on `min(time-to-next-milestone, K)`
where K is the adaptive safety-net interval.

**Adaptive K:**

- **V1 default — linear ramp**: `K(t) = K_min + (t / T_total) *
  (K_max - K_min)`. Smaller early, larger later. Simple and
  independent of E2.
- **V2 option — competence-adaptive**: `K(t) = K_min + clip(E2(t), 0,
  1) * (K_max - K_min)`. Self-calibrating — low competence keeps K
  small, high competence lengthens it.
- **Explicit stepwise schedule** — designer can override. Most
  transparent, least adaptive.

Starting numbers: `K_min=2, K_max=10, T_total=total_epochs` — tune
empirically (ADR 0011).

## Consequences

- Milestones fire eval as soon as they happen — the "interesting
  moment" always gets checked.
- K safety-net ceiling bounds the worst-case gap between evals —
  addresses stuck-agent concern.
- Denser evals early matches the intuition that early training has
  more per-epoch change per unit compute.
- Slightly harder to reason about than fixed K; mitigated by logging
  `adaptive_k` to wandb each eval so designers can audit.

## Related

- [0005](0005-frozen-evaluation.md) what eval computes
- [0007](0007-revision-cadence.md) revision subsets eval events
- [0011](0011-starting-trigger-values.md) starting values for K_min/K_max
