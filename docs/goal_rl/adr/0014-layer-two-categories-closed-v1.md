# ADR 0014 — Layer-2 categories closed in V1, opens in V2

- **Date:** 2026-04-17
- **Status:** accepted

## Context

Should the LLM be able to invent a new Layer-2 category mid-run —
e.g. `trainer_engagement`, `speedrun_pace` — beyond the seven we seed
with? It's compelling for the "new player invents new framings" spirit
the user likes. But each new category raises validation questions:
is it meaningful? a duplicate under a new name? a reward-hack dressed
up in legitimate-sounding language?

The user chose the phased option: closed for V1, open behind a gate
in V2.

## Decision

**V1:** The seven Layer-2 categories (`progress`, `completeness`,
`mastery`, `discovery`, `efficiency`, `safety`, `diversity`) are a
closed menu. The LLM may propose weight deltas on them (ADR 0013) and
sub-criteria within them (ADR 0006) — not new category names.

**V2 (backlog):** Allow the LLM to propose new categories, but each
proposed category must pass a **designer-approval gate** (human-in-
the-loop) before it becomes a valid target for future revisions. The
gate could be automated via a second LLM call that audits the proposal
against the existing menu for overlap / gamability.

## Consequences

- Simpler validator in V1 — only `GoalCategory` membership needs
  checking.
- Closes off an abuse vector (LLM renaming its own criteria into
  "categories" to escape per-category weight bounds).
- If a designer's playstyle really needs a new category, they can add
  it to the enum between runs — it's a one-line code change.

## Related

- [0003](0003-three-layer-schema.md) schema
- [0013](0013-layer-two-weights-revisable.md) weight (not category)
  revision rules
