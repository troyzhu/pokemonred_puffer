# ADR 0005 — Three-layer frozen evaluation (E1/E2/E3)

- **Date:** 2026-04-17
- **Status:** accepted

## Context

If the training rubric drifts under the revision engine, "training
reward went up" is no longer a reliable signal — the rubric may have
gotten easier, not the agent better. This is also the main attack
surface for reward-hacking: the same LLM that proposes the rubric will
see its own rubric's scores go up and can game them.

AlphaGo / AlphaStar / Agent57 all train on shaped objectives but
evaluate on a canonical held-out signal (win rate). That discipline is
the template.

## Decision

Three evaluation layers, **completely independent of the training
rubric**:

- **E1 — Dashboard.** Vector of raw snapshot facts (`badges`,
  `pokedex_caught`, `unique_maps_visited`, `blackout_count`, …). Never
  weighted. Pareto picture.
- **E2 — Frozen scalar rubric.** Fixed `CompositeRubric` constructed at
  run start, **never** modifiable. Default: reuses the baseline
  `rubric_rl` weights as the canonical judge. Produces one comparable
  number per episode.
- **E3 — Session-target scalar rubric (optional).** Frozen per run;
  scores distance to the session's stated targets.

Enforced structurally: the revision engine has no import path to the
frozen-evaluator module.

## Consequences

- Clear answer to "is the agent actually improving?" that is immune to
  rubric drift.
- Hacking watchdog becomes possible (ADR 0006): correlate training
  reward vs E2; if they diverge, roll back.
- E2 can be used across multiple runs as a canonical benchmark (same
  number, comparable across seeds and revisions).
- V1 caveat: the frozen eval runs on snapshots from training episodes,
  not a separate fixed-seed eval pass. A V2 improvement.

## Related

- [0006](0006-constrained-llm-revision.md) revision engine (operates on
  the training rubric, not E2)
- [0015](0015-baseline.md) baseline uses E2 as primary metric
