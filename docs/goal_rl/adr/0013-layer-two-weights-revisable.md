# ADR 0013 — Layer-2 weights are revisable (bounded)

- **Date:** 2026-04-17
- **Status:** accepted

## Context

A real human player's priorities shift within a run — early game
leans on progression, late game leans on mastery or completeness.
Keeping Layer-2 weights static per run is simpler but forecloses that
behaviour. The user explicitly chose the more expressive option.

The risk: another source of drift that the E2 watchdog has to catch.

## Decision

The revision engine (ADR 0006) may propose multiplicative deltas on
Layer-2 category weights — **tighter bounds than sub-criterion
deltas**:

- Multiplicative delta per revision: **Δ ∈ [0.75, 1.33]**.
- Absolute weight: **∈ [0, 10]** (inherited from sub-criterion bounds).
- Audited against the constitution's stated priorities: a revision
  that reverses a strongly-stated priority gets extra scrutiny in the
  audit trail (flagging is cosmetic in V1; V2 may reject).

E2 watchdog (ADR 0005) catches hacks regardless of whether they
operated on sub-criteria or L2 weights.

## Consequences

- More expressive than fixed weights; changes between revisions are
  small enough to be debuggable.
- The bound gap ([0.75, 1.33]) is tighter than sub-criterion bounds
  ([0.5, 2.0]) because L2 changes are leverage-amplified (affect every
  sub-criterion beneath them).
- V2 might add a stronger constitutional-check (LLM-based) before
  applying large cumulative L2 shifts across multiple revisions.

## Related

- [0006](0006-constrained-llm-revision.md) full revision scope
- [0005](0005-frozen-evaluation.md) watchdog
