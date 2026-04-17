# ADR 0006 — Constrained-LLM revision with primitives library

- **Date:** 2026-04-17
- **Status:** accepted

## Context

Eureka (Ma et al. ICLR 2024) shows LLMs can design reward *code* that
beats human-designed rewards on many benchmarks. But free-form code
generation is maximally expressive and maximally pathological — it
invites reward-hacking and makes validation hard.

SayCan (Ahn et al. CoRL 2022) demonstrates the safer pattern: LLM
proposes, feasibility layer filters. Constitutional AI (Anthropic 2022)
shows the same idea for behaviour specs.

## Decision

The LLM may propose edits only from a **bounded vocabulary**:

- Sub-criterion weight deltas within existing categories, multiplicative
  Δ ∈ [0.5, 2.0]; absolute weight ∈ [0, 10].
- Layer-2 category weight deltas, tighter bounds Δ ∈ [0.75, 1.33]
  (ADR 0013).
- Toggle built-in criteria on/off within a category.
- **Compose new criteria from a primitives library** over
  `GameStateSnapshot` fields: `fraction`, `inverse_fraction`, `delta`,
  `ratio`, `threshold_hit`, `threshold_cross`, `first_time`,
  `increment`, `rolling_avg`.
- Propose milestone predicates from the same primitives library (ADR
  0010).
- Update a narrative note to carry context across revisions.

The LLM may **not**: write arbitrary Python / shell, invent
`GameStateSnapshot` fields, add Layer-2 categories (ADR 0014), edit
Layer-1/3, or touch E2/E3.

Every proposal passes a schema validator before being applied; the E2
watchdog (ADR 0005) can roll back any applied revision.

## Consequences

- Expressive enough to cover most design adjustments (weight rebalance,
  "reward threshold crossings of X", "track recent deltas in Y").
- Safe by construction — no novel attack surface beyond what primitives
  allow.
- Validator + bounds are code invariants, not config (so designer can't
  accidentally weaken them).
- New primitives are easy to add (extend `PRIMITIVES` + signature
  dicts); the LLM learns about them via the prompt which is generated
  from the registry.

## Related

- [0005](0005-frozen-evaluation.md) watchdog that catches hacks
- [0007](0007-revision-cadence.md) when revisions fire
- [0013](0013-layer-two-weights-revisable.md) tighter L2 bounds
- [0014](0014-layer-two-categories-closed-v1.md) no new categories in V1
