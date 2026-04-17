# ADR 0010 — Milestones learned via deltas + LLM predicates

- **Date:** 2026-04-17
- **Status:** accepted

## Context

An earlier draft of the design assumed a hardcoded milestone list
(`badge_acquired`, `hm_obtained`, `region_transition`, …). The user
pushed back: that assumes we already know the game. A new player
discovers what counts as a milestone *during play* — "wait, that felt
important." Hardcoding also doesn't generalise beyond Pokémon Red.

## Decision

Two-part learned milestone detection:

1. **Delta-based auto-detection (always on).** At each eval tick,
   compare current vs previous-eval snapshot. A milestone candidate
   fires if any discrete `GameStateSnapshot` field satisfies
   `first_time` (first non-zero in the run) or `increment` (strict
   increase since last eval). Runs without any LLM call.
2. **LLM-proposed milestone predicates.** The LLM can, during a
   revision (ADR 0006), add named predicates composed from the same
   primitives library used for rubric criteria. These are validated
   identically. Example: "first time all party Pokémon are above
   level 20" as `threshold_hit(field=max_level_sum, thresh=120)`.

Both sources merge into the milestone list consumed by the trigger
controller (ADR 0007/0008).

## Consequences

- Generalises beyond Red — port to a different game, implement a new
  `GameStateSnapshot`, milestones work without re-hardcoding.
- Noise early in training (many first-times) is real; cooldown in
  trigger controller rate-limits milestone-driven evals.
- LLM predicates allow subjective milestones ("beat a tough trainer"
  encoded via a clever primitive composition) the auto-detector can't
  catch.
- Audit trail records both the source and the reason each milestone
  fired, so sifting through logs is tractable.

## Related

- [0006](0006-constrained-llm-revision.md) primitives library shared
  with rubric criteria
- [0007](0007-revision-cadence.md) milestones are a fire condition
