# ADR 0016 — V1 implementation landed

- **Date:** 2026-04-17
- **Status:** accepted (implementation)

## Context

ADRs 0001–0015 describe the design. This ADR records *what shipped*
in V1 so future maintainers can line the code up against the
decisions.

## Decision

V1 implementation:

- **Module** `pokemonred_puffer/goal_rl/` (7 files):
  - `primitives.py` — primitives library + validator + auto-milestone
    detection (ADRs 0006, 0010).
  - `schema.py` — Layer-1/2/3 dataclasses + constitution parser with
    heuristic fallback (ADRs 0003, 0004, 0009).
  - `goal_manager.py` — active state, composes `CompositeRubric`,
    apply/rollback revisions (ADRs 0003, 0006).
  - `triggers.py` — hybrid eval + revision cadence with adaptive K
    (ADRs 0007, 0008, 0010).
  - `evaluator_frozen.py` — E1/E2/E3 + hacking watchdog (ADR 0005).
  - `revision_engine.py` — LLM proposer + JSON parser + validator
    (ADR 0006).
  - `goal_grpo.py` — `GoalGRPO(CleanGRPO)` wiring all of the above
    (ADR 0001).
- **Config** — new `goal_rl:` section in `config.yaml` with defaults
  from ADR 0011.
- **CLI** — `train-goal` command in `pokemonred_puffer/train.py`.
- **Tests** — `tests/test_goal_rl.py`, 37 passing tests covering
  primitives, schema, goal manager, revision parser, triggers,
  evaluator + watchdog.
- **Docs** — `docs/goal_rl.md` + this ADR tree.

## Not in V1

- Competence-adaptive K (ADR 0008 V2 option).
- Open Layer-2 categories with approval gate (ADR 0014).
- Layer-3 constraint enforcement at the environment level (V2).
- Dedicated fixed-seed eval pass separate from training episodes
  (ADR 0005 caveat).
- Voyager-style skill-library / catastrophic-forgetting mitigation.
- Dual-mode revision (fast reweighting + slow deep revision).

## Consequences

- The design can be validated by running `train-goal` on a real ROM;
  no design-spec-vs-code drift at this moment.
- Future ADRs should reference file paths + ADR numbers so we can
  keep this alignment visible.

## Related

- All of ADRs 0001–0015 — this one is their realisation.
