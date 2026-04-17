# C02 — Frozen Evaluation Protocol

**Code:** `pokemonred_puffer/goal_rl/evaluator_frozen.py`
**Status:** adopted ([ADR 0005](../adr/0005-frozen-evaluation.md))

A ruler that doesn't move. Measures real progress independent of the
revising training rubric and catches LLM reward-hacking.

## Three layers

- **E1 — Dashboard** (never weighted): vector of raw snapshot facts
  (`badges`, `caught_pokemon_count`, `unique_maps_visited`,
  `blackout_count`, …). Pareto picture of what's improving. The canonical
  list lives in `evaluator_frozen.DASHBOARD_FIELDS` and is validated
  against the doc by `scripts/check_docs.py`.
- **E2 — Frozen scalar rubric**: a single `CompositeRubric`, fixed at
  run start, *never* editable — enforced by living in a module the
  revision engine has no import path to. Default: reuses the baseline
  `rubric_rl` rubric as the canonical judge (see
  [ADR 0015](../adr/0015-baseline.md) for why that choice).
- **E3 — Session-target rubric** (optional): frozen per run; scores
  distance-to-target under the session's Layer-2 weights.

## Cadence

Hybrid — see [C04](C04-revision-triggers-and-safety-nets.md). Eval fires
on `min(time-to-next-milestone, K)` where K is the adaptive safety-net
interval. Each eval runs M ≈ 16–32 episodes with a fixed seed set. Pure
readout — does not influence training.

## Hacking watchdog

Rolling correlation between training-rubric reward and E2. If training
reward rises ≥ `training_rise_threshold` while E2 drops ≥
`e2_drop_threshold` over the last `watchdog_window` eval windows, the
latest revision is rolled back and logged. Precedent:
[R17 AlphaStar](../research/rl-techniques.md#r17-alphastar) and
[R11 Agent57](../research/rl-techniques.md#r11-agent57) — train on
shaped objectives, evaluate on a canonical signal. The novel bit here
is using the frozen rubric as a safety net against the *revision
engine*, not just against the policy.

## Design role

Ground truth for all progress claims; without this, the system can't
be trusted.

## Related

- [C03](C03-constrained-llm-revision-engine.md) — the watchdog gates
  what the LLM-driven engine does
- [C04](C04-revision-triggers-and-safety-nets.md) — plateau on E2 is a
  revision trigger
- [R17 AlphaStar](../research/rl-techniques.md#r17-alphastar),
  [R11 Agent57](../research/rl-techniques.md#r11-agent57) — precedent
