# C07 — Learned Milestone Detection

**Code:** `pokemonred_puffer/goal_rl/primitives.py::detect_milestones`,
`pokemonred_puffer/goal_rl/triggers.py`
**Status:** adopted ([ADR 0010](../adr/0010-learned-milestones.md))

Milestones are *discovered* during play, not enumerated a priori.
Mimics a new player noticing "wait, that felt important" rather than a
speedrunner who has memorized the game.

## Two mechanisms

### 1. Delta-based auto-detection (always on)

At every eval point, compare current snapshot to the previous eval
snapshot. A milestone candidate fires if any discrete field in
`AUTO_MILESTONE_FIELDS` satisfies:

- `first_time(field)` — first non-zero in the run (e.g. first badge,
  first HM, first Pokémon caught).
- `increment(field)` — strictly increased since last eval (e.g. badges
  went up, new map visited).

Runs with zero LLM calls. Works out-of-the-box for any game with a
`GameStateSnapshot` that has meaningful discrete counters.

### 2. LLM-proposed predicates (optional)

The LLM can, during a revision
([C03](C03-constrained-llm-revision-engine.md)), add named predicates
composed from the same primitives library used for rubric criteria.
Validated identically — same safety guarantees.

Example:

```json
{
  "name": "first_balanced_team",
  "description": "All 6 party slots filled.",
  "primitive_call": {"primitive": "threshold_hit",
                     "args": {"field": "party_count", "thresh": 6}}
}
```

## Integration

The trigger controller
([C04](C04-revision-triggers-and-safety-nets.md)) asks both sources
each eval-check cycle. If any milestone fires, it's reported up to the
eval/revision path and recorded in the audit trail with its source
("auto" vs "llm") and the reason.

## Design role

The "what counts as a significant event" contract. Generalises beyond
Pokémon Red — port to another game, implement a new
`GameStateSnapshot`, milestones work without re-hardcoding.

## Risks and mitigations

- **Noise early in training.** Many fields hit `first_time` in the
  first few episodes. Trigger cooldown
  ([C04](C04-revision-triggers-and-safety-nets.md)) rate-limits
  milestone-driven revisions.
- **Missed subjective events.** Objectively small deltas (e.g. "beat
  a tough trainer") may feel important but not register. LLM
  predicates can encode these after the first revision fires on an
  auto-detected milestone.

## Related

- [C03](C03-constrained-llm-revision-engine.md) — primitives library
  shared with rubric criteria
- [C04](C04-revision-triggers-and-safety-nets.md) — milestones fire
  the event clock
- [P09 Curiosity / information-gap](../research/psychology.md#p09-curiosity-and-information-gap)
  — milestones ≈ information-gap closures
