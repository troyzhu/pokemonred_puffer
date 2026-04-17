# ADR 0009 — Constitution parsing pipeline

- **Date:** 2026-04-17
- **Status:** accepted

## Context

ADR 0004 commits to a free-text constitution. But the runtime needs
structured fields (Layer-2 weights, Layer-3 constraint list) and the
LLM revision prompt benefits from both the raw text *and* the parsed
structure. We also want the system to work without an LLM configured
(e.g. for smoke tests or reproducible CI).

## Decision

One-time parse at run start:

1. **Primary path — LLM parser.** If an LLM client is configured, call
   it with a structured-output prompt that extracts
   `playstyle, target_badges, weights, constraints, notes`. The raw
   text is preserved on the `Constitution` dataclass.
2. **Fallback — heuristic keyword parser.** If the LLM call fails or
   no client is configured, a deterministic regex/keyword parser
   detects `nuzlocke`, `speedrun`, `pokedex`, `monotype <type>`,
   `N badges`, and maps to Layer-2 weights from a canned table.

Both paths return a `GoalRLConfig` with the constitution text + the
parsed fields.

## Consequences

- Smoke tests run without LLM credentials.
- LLM-less mode is deterministic and reproducible (useful for
  matched-seed experiments — ADR 0012).
- If the LLM output is malformed, the heuristic fallback still
  produces a working config.
- The LLM parser is called **once** per run — not in the training
  loop — so its latency/cost doesn't matter for throughput.

## Related

- [0004](0004-constitution-concept.md) concept
- [0006](0006-constrained-llm-revision.md) revision engine uses the
  raw text + structure on every call
- [0012](0012-experiment-design.md) matched-seed experiments benefit
  from the heuristic path
