# C03 — Constrained-LLM Revision Engine

**Code:** `pokemonred_puffer/goal_rl/revision_engine.py`,
`pokemonred_puffer/goal_rl/primitives.py`
**Status:** adopted ([ADR 0006](../adr/0006-constrained-llm-revision.md))

The LLM proposes; a validator disposes. Expressive within a bounded
vocabulary; safe by construction.

## What the LLM may propose

- **Sub-criterion weight deltas** in existing categories. Bounded
  multiplicative `Δ ∈ [0.5, 2.0]`; absolute weight `∈ [0, 10]`.
- **Layer-2 category weight deltas**. Tighter bounds
  `Δ ∈ [0.75, 1.33]` per revision — see
  [ADR 0013](../adr/0013-layer-two-weights-revisable.md) for why
  tighter.
- **Toggles** of built-in criteria (`"category:crit_name": true/false`).
- **New criteria** composed from the **primitives library**:

  ```
  fraction(field, max)            → min(x / max, 1.0)
  inverse_fraction(field, max)    → 1.0 - min(x / max, 1.0)
  delta(field)                    → x - x_at_prev_eval
  ratio(field_a, field_b)         → a / (b + eps)
  threshold_hit(field, thresh)    → 1.0 if x >= thresh else 0.0
  threshold_cross(field, thresh)  → 1.0 if prev < thresh <= cur else 0.0
  first_time(field)               → 1.0 iff field>0 and no earlier history had >0
  increment(field)                → 1.0 iff cur > prev
  rolling_avg(field, window)      → trailing mean across history tail
  ```

  The **canonical list** lives in `primitives.PRIMITIVES` and is
  validated against the doc by `scripts/check_docs.py`.

- **New milestone predicates** from the same primitives library — see
  [C07](C07-learned-milestone-detection.md).
- A short **narrative note** carried forward as context to the next
  revision.

## What the LLM may NOT do

- Write arbitrary Python/shell (there is no eval path).
- Reference snapshot fields outside `GameStateSnapshot`.
- Add a new Layer-2 category (closed menu in V1; see
  [ADR 0014](../adr/0014-layer-two-categories-closed-v1.md)).
- Touch Layer-1 or Layer-3.
- Touch the frozen evaluator (enforced structurally — separate module).

## Pipeline

1. Trigger fires ([C04](C04-revision-triggers-and-safety-nets.md)).
2. Context bundle prepared: constitution, Layer-1/2/3 config, recent
   training-rubric reward, dashboard deltas, E2 trajectory.
3. LLM returns JSON: `{rationale, narrative, layer_two_deltas,
   added_criteria, removed_criteria, toggled_criteria,
   milestone_predicates}`.
4. Validator schema-checks, bounds-checks, rejects if any primitive
   references undefined field.
5. Apply or reject; archive the old rubric with timestamp.
6. Post-revision monitor
   ([C02 watchdog](C02-frozen-evaluation-protocol.md)).

## Precedents

- [R14 SayCan](../research/rl-techniques.md#r14-saycan) — LLM proposes,
  feasibility layer filters.
- [R18 Constitutional AI](../research/rl-techniques.md#r18-constitutional-ai)
  — LLM works within a written behavior spec.
- [R13 Eureka](../research/rl-techniques.md#r13-eureka) — LLM generates
  reward *code*; we constrain to a primitives library instead of open
  code, explicitly for safety.

## Design role

The "self-revising" engine. Every revision is auditable.

## Related

- [C02](C02-frozen-evaluation-protocol.md) eval + watchdog
- [C04](C04-revision-triggers-and-safety-nets.md) triggers
