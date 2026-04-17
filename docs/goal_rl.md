# Goal-Setting RL for Pokemon Red

A self-revising, goal-setting layer on top of the `rubric_rl` GRPO trainer. The
designer writes a short constitution describing what the agent should care
about; the system parses it into a weighted value hierarchy, runs GRPO with
that rubric, and periodically revises the rubric — under safety constraints —
based on learned milestones, plateau detection, and a frozen evaluator that
cannot be gamed.

Parallel to the existing `rubric_rl` system (see `docs/rubric_rl.md`), but with
an outer loop that adapts the reward shape during training rather than taking
it as fixed input.

> **Reading order.** Skim *Overview → Architecture → Quick Start → How It
> Works* for the whole picture. Drop into *Key Concepts* for a specific
> component, *How to Extend* for recipes. Long-form material lives in the
> [`docs/goal_rl/`](goal_rl/README.md) subdirectory: per-concept deep dives
> under [`concepts/`](goal_rl/concepts/README.md), decision history under
> [`adr/`](goal_rl/adr/README.md), literature notes under
> [`research/`](goal_rl/research/README.md).

---

## Overview

The baseline `rubric_rl` system trains on a **static**, manually-weighted
rubric. That works once the designer already knows what "good play" looks
like. This system targets the open-ended case — Pokémon is not one game but
many playstyles (story run, Nuzlocke, speedrun, Pokédex completion,
monotype, etc.), and the designer may not know in advance which criteria
will prove important.

Instead of fixing the reward, we:

1. Let the designer state intent in **free text** (the *constitution*).
2. Parse that into a **three-layer value schema**: abstract core values,
   weighted session goals, and rule-set constraints.
3. Run GRPO with a `CompositeRubric` derived from that schema.
4. On a **hybrid cadence** — milestones when they happen, adaptive K-epoch
   safety-net otherwise — run a **frozen evaluator** that the rubric itself
   cannot touch, and call an LLM to propose bounded edits to the training
   rubric.
5. **Validate and apply** the edits. Watch for reward-hacking by correlating
   training reward to the frozen evaluator and **roll back** revisions that
   look like gaming.

The result is a trainer that steers its own reward shape toward what the
designer asked for, while remaining auditable and resistant to the LLM
gaming its own rubric.

See [`goal_rl/adr/`](goal_rl/adr/README.md) for the history of how each of
these choices was reached.

---

## Motivation

Limitations of the baseline PPO + `rubric_rl` GRPO approaches:

| Problem | Baseline PPO | `rubric_rl` GRPO | Goal-setting (this module) |
|---|---|---|---|
| Reward specification | ~30 hand-tuned weights (opaque) | 4 categories × weights (interpretable) | Free-text constitution → structured hierarchy |
| Openness to playstyles | One tuned reward per setting | Same | Constitution captures Nuzlocke / speedrun / completionist |
| Milestones | Hardcoded `REQUIRED_EVENTS` | Hardcoded | Discovered from snapshot deltas + LLM predicates |
| Reward drift | N/A (static) | N/A (static) | Revisable, but bounded + watchdog-gated |
| Cheating the reward | N/A | N/A | Frozen evaluator + rollback detects hacks |
| Signal when stuck | Stuck | Stuck | Safety-net eval ceiling; plateau-triggered revision |

The three-layer schema, frozen evaluator, learned milestones, and
bounded revision engine together aim to mimic how a thoughtful new player
navigates an unknown game — set intent, watch progress, revise when stuck,
without losing sight of the original goal.

---

## Architecture

```
                            ┌───────────────────────────────┐
                            │   config.yaml (goal_rl: …)    │
                            │   constitution (free text)    │
                            └──────────────┬────────────────┘
                                           │  parsed once at run start
                                           ▼
                            ┌───────────────────────────────┐
                            │   schema.build_config         │
                            │   GoalRLConfig (L1 / L2 / L3) │
                            └──────────────┬────────────────┘
                                           │
              ┌────────────────────────────┼──────────────────────────────┐
              │                            │                              │
              ▼                            ▼                              ▼
   ┌────────────────────┐       ┌─────────────────────┐         ┌──────────────────────┐
   │   GoalManager      │       │  TriggerController  │         │   FrozenEvaluator    │
   │   ──────────────   │       │  ─────────────────  │         │   ────────────────   │
   │  composes active   │       │ adaptive K schedule │         │  E1 dashboard (raw)  │
   │  CompositeRubric   │       │ milestone detect    │         │  E2 scalar (frozen)  │
   │  apply/rollback    │       │ plateau detect      │         │  E3 session target   │
   │  revisions         │       │ revision cooldown   │         │  never revisable     │
   └──────┬─────────────┘       └──────────┬──────────┘         └──────────┬───────────┘
          │                                │                               │
          │ get_rubric()                   │ events                        │ E2 trajectory
          ▼                                ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   GoalGRPO                                      │
│                        (subclass of rubric_rl.CleanGRPO)                        │
│                                                                                 │
│   per epoch:                                                                    │
│     1. collect_episodes + train     (inherited)                                 │
│     2. eval trigger?  (milestone OR K-safety-net)                               │
│     3. if yes:  E1/E2/E3 over snapshots                                         │
│     4. HackingWatchdog.record(training_reward, E2) → maybe rollback             │
│     5. revision trigger?  (milestone OR plateau OR ceiling)                     │
│     6. RevisionEngine.propose(…) → LLM → JSON → validator                       │
│     7. GoalManager.apply_revision(…)                                            │
│     8. append audit-trail entry (JSONL)                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
          │                                                                     ▲
          │ proposes bounded edits                                               │
          ▼                                                                     │
   ┌──────────────────────────────────────────────────────────────────────────────┴───────┐
   │                                RevisionEngine                                       │
   │                                                                                     │
   │   build_prompt ─► LLM call ─► _extract_json ─► parse_revision ─► RevisionProposal   │
   │                                                                                     │
   │   validated against:                                                                │
   │     - PRIMITIVES library (primitives.py)                                            │
   │     - GoalCategory enum (closed V1 menu)                                            │
   │     - weight bounds (L2 Δ ∈ [0.75, 1.33]; sub-criterion weight ∈ [0, 10])           │
   └─────────────────────────────────────────────────────────────────────────────────────┘
```

**Files.** See [Code Map](#code-map) below for the file-to-concept table.

---

## Code Map

| File | Concept | Key symbols |
|---|---|---|
| `pokemonred_puffer/goal_rl/__init__.py` | Module entry | — |
| `pokemonred_puffer/goal_rl/primitives.py` | [C03](goal_rl/concepts/C03-constrained-llm-revision-engine.md), [C07](goal_rl/concepts/C07-learned-milestone-detection.md) — primitives library | `PRIMITIVES`, `PrimitiveCall`, `validate_primitive_call`, `detect_milestones` |
| `pokemonred_puffer/goal_rl/schema.py` | [C01](goal_rl/concepts/C01-three-layer-value-schema.md), [C08](goal_rl/concepts/C08-constitution-parsing.md) — three-layer schema + constitution parser | `GoalCategory`, `LayerOne/Two/Three`, `Constitution`, `GoalRLConfig`, `build_config`, `parse_constitution` |
| `pokemonred_puffer/goal_rl/goal_manager.py` | [C01](goal_rl/concepts/C01-three-layer-value-schema.md), [C03](goal_rl/concepts/C03-constrained-llm-revision-engine.md) — active state + rubric composition | `GoalManager`, `CriterionSpec`, `RevisionRecord` |
| `pokemonred_puffer/goal_rl/triggers.py` | [C04](goal_rl/concepts/C04-revision-triggers-and-safety-nets.md), [C07](goal_rl/concepts/C07-learned-milestone-detection.md) — cadence decisions | `TriggerConfig`, `TriggerController`, `MilestonePredicate` |
| `pokemonred_puffer/goal_rl/evaluator_frozen.py` | [C02](goal_rl/concepts/C02-frozen-evaluation-protocol.md) — frozen eval + watchdog | `FrozenEvaluator`, `HackingWatchdog`, `EvalBatch`, `dashboard_from_snapshot`, `DASHBOARD_FIELDS` |
| `pokemonred_puffer/goal_rl/revision_engine.py` | [C03](goal_rl/concepts/C03-constrained-llm-revision-engine.md) — LLM proposer + validator | `RevisionEngine`, `RevisionProposal`, `parse_revision`, `build_prompt` |
| `pokemonred_puffer/goal_rl/goal_grpo.py` | [C05](goal_rl/concepts/C05-integration-with-rubric-rl.md) — integrated trainer | `GoalGRPO`, `GoalRLRuntimeConfig` |
| `config.yaml` | runtime config | `goal_rl:` section |
| `pokemonred_puffer/train.py` | CLI | `train_goal` command |
| `tests/test_goal_rl.py` | unit tests | 37 tests |
| `scripts/check_docs.py` | doc-drift guards | primitives / dashboard / link checks |

---

## Quick Start

### Install

This module uses the same env as the rest of `pokemonred_puffer`. If you
have the conda env set up:

```bash
conda activate pokemonred   # or your equivalent
```

### Minimal run

Defaults use the heuristic constitution parser (no LLM call at run start)
and dry-run revisions (no LLM call during training). Safe to run without
credentials:

```bash
python -m pokemonred_puffer.train train-goal \
    --debug \
    --vectorization serial \
    --dry-run-revisions \
    --constitution "Story run aiming for 4 badges, careful team"
```

This exercises the full pipeline (parsing, trigger scheduling, frozen
evaluation, audit logging) without making any paid LLM calls.

### With LLM revisions enabled

Set your Anthropic API key and remove `--dry-run-revisions`:

```bash
export ANTHROPIC_API_KEY=sk-…
python -m pokemonred_puffer.train train-goal \
    --constitution "Nuzlocke run, aim for 5 badges, balanced team"
```

The model is configured in `config.yaml::goal_rl.revision_engine.model`.

### Example constitutions

| Playstyle | Example string |
|---|---|
| **Story run** | `"Beat the game, moderate risk, aim for all 8 badges."` |
| **Nuzlocke** | `"Nuzlocke run. 5 badges target. Safety first — never lose a Pokémon."` |
| **Pokédex hunter** | `"Catch as many Pokémon as possible. Progress is secondary to completeness."` |
| **Speedrunner** | `"Speedrun to Elite Four. Efficiency is everything."` |
| **Monotype** | `"Monotype electric team, explore thoroughly, 4 badges."` |

The heuristic parser recognizes keywords (`nuzlocke`, `speedrun`, `pokedex`,
`monotype electric`, `N badges`) and seeds the Layer-2 weights accordingly.
An LLM parser (configured via the same `goal_rl.revision_engine` block)
will produce a richer extraction on ambiguous cases.

### Output artifacts

- **Audit log**: JSONL at `goal_rl.runtime.audit_log_path` (default:
  `goal_rl_audit.jsonl`). One line per eval, revision attempt, or rollback.
- **Wandb logs**: `goal_rl/*` namespace — `e2_mean`, dashboard fields,
  `num_revisions_applied`, `adaptive_k`, etc.
- **Checkpoints**: inherited from `CleanGRPO`.

---

## How It Works — Step by Step

Per epoch (inside `GoalGRPO.run_epoch`):

1. **Tick the trigger clock.** `TriggerController.tick()` advances the
   epoch counter; the adaptive K is a function of it.
2. **Refresh the training rubric** from `GoalManager.get_rubric()`. If the
   previous epoch applied a revision, this epoch trains against the new
   shape.
3. **Collect episodes and train** via the inherited `CleanGRPO.run_epoch()`
   — GRPO with the current rubric as the reward.
4. **Record every episode snapshot** to `goal_manager.record_snapshot()`
   so history-dependent primitives (like `first_time`) work.
5. **Check eval trigger.** Auto-detect milestones (`primitives.detect_milestones`
   — `first_time` / `increment` over `GameStateSnapshot`) + evaluate any
   LLM-proposed milestone predicates. If a milestone fires, eval; otherwise
   eval if adaptive K epochs have elapsed since the last one.
6. **Frozen evaluation** (if triggered). `FrozenEvaluator` scores the
   episode snapshots against the E2 rubric (never-edited) and emits the E1
   dashboard.
7. **Hacking watchdog.** `HackingWatchdog.record(training_reward, E2)`. If
   training reward has risen but E2 has dropped over the recent window, the
   last revision is rolled back and the watchdog history reset.
8. **Check revision trigger** (subset of eval events): milestone-driven,
   plateau-on-E2, or revision-ceiling. Cooldown enforces a minimum gap
   between revisions.
9. **Revision engine** (if triggered). `RevisionEngine.propose(...)` builds
   a prompt, calls the LLM, parses the JSON, and validates each proposed
   edit against the primitives library and weight bounds. Malformed entries
   are dropped with warnings; catastrophically bad responses fall back to
   an empty proposal (no-op).
10. **Apply or reject.** `GoalManager.apply_revision(...)` applies the
    validated edits within additional runtime bounds (Layer-2 deltas
    Δ ∈ [0.75, 1.33], sub-criterion weights ∈ [0, 10], no shadowing of
    existing criterion names).
11. **Audit.** Every eval, revision attempt, rollback, and hacking event is
    written to the JSONL audit log and logged to wandb.

The next epoch starts again from step 1 — the rubric seen at step 2 is now
whatever came out of step 10.

---

## Configuration Reference

`config.yaml::goal_rl` section:

```yaml
goal_rl:
  constitution: ""               # free-text; empty = equal weights, no constraints
  triggers:
    k_min: 2                     # adaptive K starting value (epochs)
    k_max: 10                    # adaptive K ceiling (epochs)
    total_epochs: 100            # ramp length for linear-K schedule
    plateau_epsilon: 0.02        # E2 gain below this counts as flat
    plateau_window: 3            # eval windows of flatness to declare a plateau
    revision_ceiling_epochs: 30  # force revision after this long without one
    revision_cooldown_epochs: 3  # minimum epochs between revisions
  evaluator:
    e2_rubrics: null             # optional override of E2 rubric weights
    watchdog_window: 3
    watchdog_training_rise: 0.05
    watchdog_e2_drop: 0.05
  revision_engine:
    provider: anthropic          # or openai
    model: claude-haiku-4-5-20251001
    max_tokens: 1200
    dry_run: false               # true = never call the LLM
  runtime:
    audit_log_path: goal_rl_audit.jsonl
    max_llm_revisions: null      # null = bounded only by cooldown + ceiling
```

All runtime bounds (Layer-2 delta limits, sub-criterion weight bounds) are
**code constants** in `goal_manager.py` (`LAYER_TWO_DELTA_MIN/MAX`,
`SUB_WEIGHT_MIN/MAX`), not config, by design — they are safety invariants
rather than tunables.

---

## Key Concepts

Each concept has a stable ID (`C##`) used throughout this doc and the
Python module docstrings. Full-length treatments live in the
[`concepts/`](goal_rl/concepts/README.md) subdirectory; short summaries
follow.

- **[C01 — Three-layer value schema](goal_rl/concepts/C01-three-layer-value-schema.md).**
  L1 core values (SDT), L2 weighted session goals (closed 7-category menu),
  L3 constraints (rule-sets, not goals). Open-ended playstyles expressible
  as weighted combinations.
- **[C02 — Frozen evaluation protocol](goal_rl/concepts/C02-frozen-evaluation-protocol.md).**
  E1 dashboard + E2 frozen scalar + optional E3 session target. Never
  editable by the revision engine. The watchdog correlates training reward
  with E2 and rolls back revisions that hack the training rubric.
- **[C03 — Constrained-LLM revision engine](goal_rl/concepts/C03-constrained-llm-revision-engine.md).**
  LLM proposes edits from a bounded vocabulary (primitives library +
  weight deltas + toggles). Validator rejects anything off-schema. Safe
  by construction.
- **[C04 — Revision triggers & safety nets](goal_rl/concepts/C04-revision-triggers-and-safety-nets.md).**
  Hybrid eval cadence: milestones fire immediately; adaptive K-epoch
  safety net otherwise. Revision fires on milestone OR plateau OR
  revision-ceiling, gated by cooldown.
- **[C05 — Integration with `rubric_rl`](goal_rl/concepts/C05-integration-with-rubric-rl.md).**
  `GoalGRPO` subclasses `CleanGRPO`; refreshes `self.rubric` from the goal
  manager each epoch; the static-rubric path stays intact.
- **[C06 — Design tensions](goal_rl/concepts/C06-design-tensions.md).**
  Stability vs responsiveness, expressiveness vs trainability, compute,
  catastrophic forgetting, human-oversight bottleneck. Mitigations
  catalogued.
- **[C07 — Learned milestone detection](goal_rl/concepts/C07-learned-milestone-detection.md).**
  Delta-based auto-detection + LLM-proposed predicates. No hardcoded
  milestone list; generalises beyond Pokémon Red.
- **[C08 — Constitution parsing](goal_rl/concepts/C08-constitution-parsing.md).**
  Free-text input → structured fields via one LLM call (or heuristic
  fallback). Raw text + structure both retained at runtime.

---

## How to Extend

### Add a new primitive

Needed when the LLM keeps wanting a measurement it can't express with the
existing library. **Procedure:**

1. Add the function to `goal_rl/primitives.py`:
   ```python
   def my_primitive(ctx: PrimitiveContext, field: str, ...) -> float:
       x = _get_numeric(ctx.snapshot, field)
       return ...  # must return a float; [0, 1] for criteria
   ```
2. Register in `PRIMITIVES` (name → function) and in
   `PRIMITIVE_SIGNATURES` (required kwargs + which ones are snapshot
   fields):
   ```python
   PRIMITIVES["my_primitive"] = my_primitive
   PRIMITIVE_SIGNATURES["my_primitive"] = {
       "fields": {"field"}, "required": {"field", "<other_args>"}
   }
   ```
3. Export from `__all__`.
4. Add a test in `tests/test_goal_rl.py::TestPrimitives`.
5. Add a line to the primitives list in
   [`C03`](goal_rl/concepts/C03-constrained-llm-revision-engine.md) (the
   doc-drift check in `scripts/check_docs.py` will fail otherwise).

The validator picks it up automatically. The LLM discovers it via the
prompt — `build_prompt` lists primitives from `PRIMITIVE_SIGNATURES`.

### Add a built-in criterion to an existing category

Modify the relevant factory in `goal_rl/goal_manager.py` (e.g.
`_progress_rubric`). Remember to add the criterion's name to the toggle
dict by running the factory once — `_builtin_enabled` is built from the
factory output.

### Change the E2 (frozen) rubric

Either pass an explicit weights dict via
`config.yaml::goal_rl.evaluator.e2_rubrics` (parsed by
`build_e2_rubric_from_config`) or modify
`rubric_rl.rubrics.DEFAULT_RUBRIC_WEIGHTS` — E2 defaults to the latter.

**Note:** changing the E2 rubric retrospectively invalidates comparison
across runs. Freeze your choice before an experiment and note it in the
run's wandb config.

### Add a new constraint kind (Layer 3)

Currently constraints are declared but only surface in the LLM prompt and
audit trail — enforcement at the environment level is a V2 concern.

1. Add a variant to `ConstraintKind` in `goal_rl/schema.py`.
2. (Optional) Teach the heuristic parser to detect it
   (`_heuristic_parse`).
3. (Enforcement, V2) Hook into `RubricRewardEnv` to penalize transitions
   that violate the constraint. Not implemented in V1.

### Add a new Layer-2 category

**Deliberately difficult in V1** (by design — prevents LLM from inventing
plausible-sounding categories mid-run). To add one:

1. Add a variant to `GoalCategory` in `goal_rl/schema.py`.
2. Add a description to `CATEGORY_DESCRIPTIONS`.
3. Add a factory `_xxx_rubric()` and entry in `_BUILTIN_FACTORIES` in
   `goal_rl/goal_manager.py`.
4. (Optional) Add heuristic weighting in `_heuristic_parse`.

### Tune trigger thresholds

Start with the defaults (`ε=0.02`, `N=3`, `K_min=2`, `K_max=10`). After a
first real run, adjust based on:

- If plateaus never fire → lower `plateau_epsilon` or tighten
  `plateau_window`.
- If plateaus fire too often → raise `plateau_epsilon`.
- If evals feel sparse late in training → lower `k_max` or shorten
  `total_epochs`.
- If the LLM spends too much → raise `revision_cooldown_epochs` or set
  `max_llm_revisions`.

### Writing a new ADR

When you make a design decision worth preserving:

1. `ls docs/goal_rl/adr/ | sort | tail -3` to pick the next number.
2. Copy an existing ADR's frontmatter; fill in Context / Decision /
   Consequences / Related.
3. Add a row to [`adr/README.md`](goal_rl/adr/README.md).
4. Add a one-line entry to the [Decision Log](#decision-log) below with a
   link to the ADR.

---

## Testing & Verification

### Unit tests

```bash
python -m pytest tests/test_goal_rl.py -v
```

37 tests covering primitives (evaluation, validation, field checks,
auto-milestone detection), schema (heuristic parser, Layer-2
canonicalization), goal manager (apply / rollback / custom criteria),
revision parser (valid / malformed / clipping), triggers (adaptive K,
plateau, cooldown), and evaluator + watchdog.

### Doc drift guards

```bash
python scripts/check_docs.py
```

Validates that the primitives list and E1 dashboard in the docs match
the code, and that all cross-references (`[C0x]`, `[S##]`, …) resolve.
Run before opening a PR; CI runs it automatically.

### CLI smoke test

```bash
python -m pokemonred_puffer.train train-goal \
    --debug --vectorization serial --dry-run-revisions \
    --constitution "Smoke test run"
```

Confirms: constitution parses, module imports, goal manager builds a
rubric, trigger controller schedules evals, frozen evaluator computes
dashboards, audit log is written. No LLM call. No ROM required at
import; will require a ROM at environment creation.

### First experiment (once you have ROM + compute)

Per [ADR 0012](goal_rl/adr/0012-experiment-design.md) and
[B03](goal_rl/research/codebase.md#b03--baseline-original-ppo):
N=3 matched seeds × {baseline PPO, goal-setting system}; identical
compute budget; compare E2 mean trajectory and E1 dashboard Pareto.

---

## Known Limitations (V1)

1. **Layer-3 enforcement is cosmetic.** Constraints are stored and shown
   to the LLM but not yet enforced by the environment. Nuzlocke-style
   rules would need hooks in `RubricRewardEnv`.
2. **Frozen eval shares training episodes.** V1 evaluates on snapshots
   collected during training rather than a separate fixed-seed eval pass.
   Good enough for most signals but not strictly apples-to-apples across
   revisions.
3. **Adaptive K is linear only.** Competence-adaptive (E2-tied) K is a V2
   feature.
4. **No skill library.** Voyager-style skill preservation is a V2
   concern.
5. **Training rubric is the only revisable surface.** We don't revise the
   policy architecture, hyperparameters, or curriculum of starting states
   (though `GRPOStateManager` supports the last of these manually).
6. **LLM cost is uncapped by default.** `max_llm_revisions: null` in
   config means "bounded only by cooldown + ceiling". Set an explicit cap
   before expensive runs.

---

## Decision Log

Append-only summary. Full context for each decision is in the
corresponding ADR file under [`goal_rl/adr/`](goal_rl/adr/README.md).

| ADR | Date | Summary |
|---|---|---|
| [0001](goal_rl/adr/0001-scope.md) | 2026-04-17 | Project scope: new `goal_rl` module layered on top of `rubric_rl`. |
| [0002](goal_rl/adr/0002-source-quality-tiers.md) | 2026-04-17 | Strict source-quality tiers for literature citations. |
| [0003](goal_rl/adr/0003-three-layer-schema.md) | 2026-04-17 | Three-layer value schema: L1 core values / L2 session goals / L3 constraints. |
| [0004](goal_rl/adr/0004-constitution-concept.md) | 2026-04-17 | Free-text constitution as designer intent. |
| [0005](goal_rl/adr/0005-frozen-evaluation.md) | 2026-04-17 | Three-layer frozen evaluation (E1/E2/E3), unreachable from revision engine. |
| [0006](goal_rl/adr/0006-constrained-llm-revision.md) | 2026-04-17 | Constrained-LLM revisions from a primitives library. |
| [0007](goal_rl/adr/0007-revision-cadence.md) | 2026-04-17 | Single revision mechanism, three fire conditions. |
| [0008](goal_rl/adr/0008-eval-cadence-adaptive-k.md) | 2026-04-17 | Hybrid eval cadence with adaptive K safety net. |
| [0009](goal_rl/adr/0009-constitution-parsing-pipeline.md) | 2026-04-17 | Constitution parsed by LLM with heuristic fallback. |
| [0010](goal_rl/adr/0010-learned-milestones.md) | 2026-04-17 | Milestones learned via deltas + LLM predicates (no hardcoded list). |
| [0011](goal_rl/adr/0011-starting-trigger-values.md) | 2026-04-17 | Starting plateau / adaptive-K / watchdog thresholds. |
| [0012](goal_rl/adr/0012-experiment-design.md) | 2026-04-17 | First experiment: N=3 matched seeds vs baseline PPO. |
| [0013](goal_rl/adr/0013-layer-two-weights-revisable.md) | 2026-04-17 | Layer-2 weights are revisable (tight bounds Δ ∈ [0.75, 1.33]). |
| [0014](goal_rl/adr/0014-layer-two-categories-closed-v1.md) | 2026-04-17 | Layer-2 categories closed in V1; opens in V2 via approval gate. |
| [0015](goal_rl/adr/0015-baseline.md) | 2026-04-17 | Baseline to beat: original repo PPO. |
| [0016](goal_rl/adr/0016-v1-implementation-landed.md) | 2026-04-17 | V1 implementation landed (module + CLI + tests). |

---

## Open Questions

1. **Compute ceiling.** Rough LLM-call budget per run. Deferred — set
   before the first real training run based on observed cadence from a
   dry-run smoke test.

---

## Research Notes (appendix)

The literature review that informed the design lives in the
[`goal_rl/research/`](goal_rl/research/README.md) subdirectory:

- [**psychology.md**](goal_rl/research/psychology.md) — P01–P10 (SDT,
  goal-setting theory, CLT, TOTE, implementation intentions,
  metacognition, flow, self-efficacy, curiosity, Rokeach values).
- [**game-studies.md**](goal_rl/research/game-studies.md) — G01–G03
  (SDT-in-games, GameFlow, Gamer Motivation Model).
- [**rl-techniques.md**](goal_rl/research/rl-techniques.md) — R01–R20
  (Options, FeUdal, HIRO, HER, UVFA, PLR, PAIRED, ICM, RND, Go-Explore,
  Agent57, Reflexion, Eureka, SayCan, ELLM, AlphaGo/Zero, AlphaStar,
  Constitutional AI, Voyager, DreamerV3).
- [**codebase.md**](goal_rl/research/codebase.md) — B01–B03 (`rubric_rl`
  overview, extension points, baseline PPO).
- [**sources.md**](goal_rl/research/sources.md) — S01–S39 numbered
  bibliography, grouped by tier.

Source-tier policy and cross-reference conventions are documented in
[`goal_rl/README.md`](goal_rl/README.md).
