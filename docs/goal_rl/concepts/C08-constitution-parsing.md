# C08 — Constitution Parsing

**Code:** `pokemonred_puffer/goal_rl/schema.py::parse_constitution`,
`build_config`
**Status:** adopted
([ADR 0004](../adr/0004-constitution-concept.md),
 [ADR 0009](../adr/0009-constitution-parsing-pipeline.md))

User writes free text; a one-time parser extracts structured fields;
runtime uses both.

## Pipeline (one LLM call at run start)

1. **Input.** Designer's free-text constitution string. Stored on the
   `Constitution` dataclass as `raw_text`.
2. **Output schema.** `{playstyle, target_badges, weights,
   constraints, notes}`. Weights and constraints map directly to
   Layer-2 weights and Layer-3 constraints in
   [C01](C01-three-layer-value-schema.md).
3. **Primary path — LLM parser.** If an LLM client is configured, call
   it with a structured-output prompt. `schema.parse_constitution`
   issues one Anthropic `messages.create` (or OpenAI-shaped) call and
   expects a JSON response.
4. **Fallback — heuristic parser.** If the LLM call fails or no client
   is configured, a deterministic keyword parser maps `nuzlocke`,
   `speedrun`, `pokedex`, `monotype <type>`, `N badges`, …, to a
   canned Layer-2 weighting.
5. **Both the raw text and the structured JSON are retained** — the
   revision engine ([C03](C03-constrained-llm-revision-engine.md))
   sees both on every call so it can reason about the designer's *why*,
   not just the parsed weights.

## Defaults

If no constitution is provided: equal Layer-2 weights across all seven
categories, no constraints. The run will behave like an open-ended
exploratory agent; not recommended for serious experiments, but useful
as a "what does this system do without any intent?" baseline.

## Why both LLM and heuristic

- **LLM** captures nuance — "try to be careful but don't waste time"
  maps to `(safety: moderate, efficiency: moderate)` better than any
  keyword matcher could.
- **Heuristic** guarantees determinism. Matched-seed experiments
  ([ADR 0012](../adr/0012-experiment-design.md)) rely on the parse
  being reproducible; running the heuristic path pins the config.
  Smoke tests work without credentials.

## Related

- [C01](C01-three-layer-value-schema.md) — the schema the constitution
  maps to
- [C03](C03-constrained-llm-revision-engine.md) — reads the raw text
  on every revision
- [R18 Constitutional AI](../research/rl-techniques.md#r18-constitutional-ai)
  — inspiration
