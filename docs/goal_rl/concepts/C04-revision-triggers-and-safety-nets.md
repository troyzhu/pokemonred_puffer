# C04 — Revision Triggers & Safety Nets

**Code:** `pokemonred_puffer/goal_rl/triggers.py`
**Status:** adopted
([ADR 0007](../adr/0007-revision-cadence.md),
 [ADR 0008](../adr/0008-eval-cadence-adaptive-k.md),
 [ADR 0011](../adr/0011-starting-trigger-values.md))

**Hybrid cadence.** Milestones fire evaluation/revision as soon as they
occur; a safety-net interval `K` guarantees "at most K epochs without
an eval" as the worst case. `K` is **adaptive** — smaller early in
training when learning is fastest, larger later.

## The two clocks

- **Event clock (milestones)** — fires whenever a milestone predicate
  flips. Milestones are discovered, not hardcoded — see
  [C07](C07-learned-milestone-detection.md). Delta-based auto-detection
  + LLM-proposed predicates.
- **Safety-net clock (K-epoch ceiling)** — hard upper bound on time
  between evals. **Effective cadence = `min(time-to-next-milestone,
  K)`.**

## Adaptive K

V1 default is a linear ramp from `k_min=2` to `k_max=10` over
`total_epochs`. A competence-adaptive form tied to E2 is a V2 option.

## Revision trigger (subset of eval events)

Every eval is a candidate revision; the actual revision fires if at
least one of these is true:

1. **Milestone trigger** — the eval was milestone-driven (almost
   always revise at milestones: the game phase has shifted).
2. **E2 plateau trigger** — E2 hasn't improved by `≥ plateau_epsilon`
   over the last `plateau_window` eval windows. Grounded in
   [P06 Nelson & Narens metacognitive monitoring](../research/psychology.md#p06-metacognitive-monitoring)
   — divergence from predicted learning.
3. **Revision ceiling** — no revision has fired in
   `revision_ceiling_epochs` training epochs. Forces at least one
   revision per ceiling even if nothing else fires.

## Debouncing

Cooldown (`revision_cooldown_epochs`) enforces a minimum gap between
revisions so the policy has time to adapt to one change before the
next. The E2 watchdog in
[C02](C02-frozen-evaluation-protocol.md) can roll back any revision
that hurts the frozen scalar.

## Related

- [C02](C02-frozen-evaluation-protocol.md) — what eval computes;
  watchdog
- [C03](C03-constrained-llm-revision-engine.md) — what a revision
  actually does
- [C07](C07-learned-milestone-detection.md) — where milestones come
  from
- [R06 Prioritized Level Replay](../research/rl-techniques.md#r06-prioritized-level-replay)
  — learning-progress-based scheduling precedent
