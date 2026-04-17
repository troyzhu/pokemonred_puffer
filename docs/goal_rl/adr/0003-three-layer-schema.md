# ADR 0003 — Three-layer value schema

- **Date:** 2026-04-17
- **Status:** accepted

## Context

Pokémon is not one game but many playstyles: story run, Nuzlocke,
monotype, speedrun, Pokédex completion, level-cap challenge. A single
flat list of "primary goals" either (a) forces the designer to pick
one, (b) blurs goals with rules ("release fainted Pokémon" is not a
goal, it's a constraint), or (c) hardcodes a specific framing.

Psychology literature supports separating:

- Abstract terminal values (Self-Determination Theory: competence +
  autonomy) — always-on anchors.
- Concrete session goals — what this run is optimising for.
- Instrumental rules — how the goals may be pursued (Rokeach terminal
  vs instrumental distinction).

## Decision

Three explicit layers:

- **Layer 1 — core values (abstract, always on):** competence + autonomy
  from SDT. Anchor LLM-prompt semantics; not directly optimised.
- **Layer 2 — session-configurable primary goals (weighted menu):**
  closed 7-category set — `progress`, `completeness`, `mastery`,
  `discovery`, `efficiency`, `safety`, `diversity`. Non-zero weights on
  conflicting goals are explicitly allowed.
- **Layer 3 — constraints (rule-sets, not goals):** Nuzlocke, monotype,
  level-cap, no-items. Penalise violations; do not provide positive
  reward. Frozen per run.

## Consequences

- Open-ended playstyles become expressible as weighted combinations
  plus constraint toggles.
- Validator has a cleaner job — goals vs constraints are structurally
  distinct.
- Adding a new playstyle = picking weights + constraint set, not
  editing the rubric factory.
- See ADRs 0013 (weights revisable), 0014 (categories closed in V1),
  and 0010 (learned milestones) for how this schema evolves at runtime.

## Related

- [0004](0004-constitution-concept.md) constitution binds user text to schema
- [0013](0013-layer-two-weights-revisable.md) L2 weight revision rules
- [0014](0014-layer-two-categories-closed-v1.md) L2 closed menu in V1
