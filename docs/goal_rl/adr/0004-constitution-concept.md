# ADR 0004 — Constitution = free-text intent (concept)

- **Date:** 2026-04-17
- **Status:** accepted

## Context

The designer needs to communicate intent to the system. A fully
structured input form (fields for each weight, each constraint) is
tedious and doesn't capture the *why*. Constitutional AI (Bai et al.
2022) demonstrates that a written natural-language specification can
guide complex model behaviour.

## Decision

The designer writes a **short free-text constitution** describing what
they want. This string is:

1. Parsed at run start into structured Layer-2/3 fields (ADR 0009).
2. **Retained verbatim** so the revision engine's LLM sees both the
   raw intent *and* the parsed structure on every revision call.

Example: "Nuzlocke run aiming for 5 badges, balanced team, prefer
exploration over efficiency."

## Consequences

- UX: designer writes one paragraph rather than filling a form.
- Runtime: LLM revision prompts are grounded in the designer's *why*,
  which reduces drift from intent.
- Requires a parser step (ADR 0009) — LLM or heuristic.
- Stored in the audit trail for every run, making experimental
  provenance explicit.

## Related

- [0009](0009-constitution-parsing-pipeline.md) parsing pipeline
- [0003](0003-three-layer-schema.md) schema the constitution maps to
