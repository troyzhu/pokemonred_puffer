# C01 — Three-Layer Value Schema

**Code:** `pokemonred_puffer/goal_rl/schema.py`
**Status:** adopted ([ADR 0003](../adr/0003-three-layer-schema.md))

Primary-goal specification splits across three layers so the system
supports open-ended, conflicting, and self-imposed playstyles.

- **Layer 1 — Core values** (abstract, always on). Competence +
  Autonomy, from Self-Determination Theory
  ([P01](../research/psychology.md#p01-self-determination-theory-sdt),
  [G01](../research/game-studies.md#g01-sdt-in-games)). Too abstract for
  direct training; anchors LLM-prompt semantics.
- **Layer 2 — Session goals** (weighted closed menu in V1).
  `progress`, `completeness`, `mastery`, `discovery`, `efficiency`,
  `safety`, `diversity`. Non-zero weights on conflicting goals are
  explicitly allowed — that's what "open-ended" means.
- **Layer 3 — Constraints** (rule-sets, not goals). Nuzlocke,
  monotype, level-cap, no-items. Don't provide reward; penalize
  violations or mask transitions. Frozen per run.

The split between L2 and L3 encodes Rokeach's
([P10](../research/psychology.md#p10-rokeach-values)) distinction
between *terminal* values (goals) and *instrumental* values (rules).

**Why three layers and not one list.** Conflating "I want to complete
the pokédex" (Layer 2 goal) with "I will release any fainted Pokémon"
(Layer 3 constraint) blurs the terminal-vs-instrumental distinction and
makes the validator harder to write. It also makes it hard to express
"Nuzlocke-style Pokédex run" as a composition of goal-set × rule-set.

**Design role.** This is the top-level schema; everything else plugs
into it.

**Related.**
[C08](C08-constitution-parsing.md) parses user text into L1/L2/L3 •
[R18](../research/rl-techniques.md#r18-constitutional-ai)
Constitutional AI (the constitution is the free-text wrapper over
L1–L3).
