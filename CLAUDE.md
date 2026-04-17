# Claude Code handoff — pokemonred_puffer

This file is read automatically by Claude Code at session start. Scan it
before doing anything substantive, especially if you're picking up from a
previous session.

## What this project is

Pokémon Red RL training library built on PufferLib. Three training paths
are available:

- **Baseline PPO** with hand-tuned reward weights — `train` CLI command;
  reward classes in `pokemonred_puffer/rewards/baseline.py`. Known to
  beat the game per the top-level [`README.md`](README.md).
- **GRPO with rubric-based rewards** — `train-grpo` CLI command. Design:
  [`docs/rubric_rl.md`](docs/rubric_rl.md).
- **Goal-setting layer on top of GRPO** — `train-goal` CLI command.
  V1 landed. Design: [`docs/goal_rl.md`](docs/goal_rl.md); concept pages,
  ADRs, and research notes live under
  [`docs/goal_rl/`](docs/goal_rl/README.md).

## Current state (as of 2026-04-17)

- **V1 of the goal_rl layer is implemented and tested** (37 passing unit
  tests). See [ADR 0016](docs/goal_rl/adr/0016-v1-implementation-landed.md)
  for the module-to-concept map.
- **Not yet exercised against a real ROM.** The intended first experiment
  (N=3 matched seeds vs baseline PPO) is spec'd in
  [ADR 0012](docs/goal_rl/adr/0012-experiment-design.md).
- **Only outstanding design question:** LLM-call compute ceiling per run —
  to be set empirically after a dry-run smoke test. See "Open Questions"
  in [`docs/goal_rl.md`](docs/goal_rl.md).
- **V2 backlog** is listed under "Not in V1" in
  [ADR 0016](docs/goal_rl/adr/0016-v1-implementation-landed.md). Key
  items: Layer-3 constraint enforcement at the env level,
  competence-adaptive K, dedicated fixed-seed eval pass, skill library.
- **Pre-existing lint issues** in `pokemonred_puffer/rubric_rl/grpo.py`
  (4 unused imports + 1 unused local) — **not introduced by goal_rl
  work**. Clean up in a separate pass if desired.

## Verification commands

```bash
# Unit tests (37 tests, ~0.4s):
python -m pytest tests/test_goal_rl.py -v

# Doc-drift guards (primitives ↔ code; dashboard ↔ code; cross-refs; link targets):
python scripts/check_docs.py

# CLI smoke test (no LLM, no ROM — exits before env creation but exercises parse/trigger/eval wiring):
python -m pokemonred_puffer.train train-goal \
    --debug --vectorization serial --dry-run-revisions \
    --constitution "Smoke test"

# Ruff (matches CI):
ruff format pokemonred_puffer
ruff check pokemonred_puffer
```

CI runs pytest + ruff format + ruff check + `scripts/check_docs.py` on
push/PR (`.github/workflows/workflow.yml`).

## Project conventions (project-wide rules, not one-off preferences)

1. **Research-first, discussion-first for novel design work.** When
   adding new ML/RL concepts (not routine bug fixes or tidying), do
   literature review → consolidate → discuss with the user → only then
   implement. Past sessions that jumped to code were explicitly
   corrected. See
   [ADR 0002](docs/goal_rl/adr/0002-source-quality-tiers.md) for the
   source-quality standard that goes with this.

2. **Strict source-quality tiers for citations.** Tier A (peer-reviewed)
   preferred; Tier B (named-lab tech reports) acceptable; Tier C (arXiv
   from a clearly named lab) used as *design inspiration only* and
   flagged as Tier C. Never cite Medium, uncurated blogs, or
   unverifiable arXiv IDs. Flag uncertain citations with `[verify]`.

3. **Append-only decision log.** When making a design decision worth
   remembering, add a new ADR under
   [`docs/goal_rl/adr/`](docs/goal_rl/adr/) (copy an existing one for
   the format) and a row to the Decision Log table in
   `docs/goal_rl.md`. ADR numbers are stable; superseded entries stay
   with `Status: superseded by NNNN`.

4. **Keep docs in sync with code.** `scripts/check_docs.py` enforces
   the primitives list and E1 dashboard stay in sync with code, and
   that all `[C##]/[S##]/[P##]/[G##]/[R##]/[B##]/[ADR ####]` refs
   resolve. Run it before commits; CI runs it too.

5. **Ask before destructive or shared-state actions.** Do **not** run
   `git commit`, `git push`, `git reset --hard`, or anything else that
   touches shared state without explicit instruction. Changes are made
   for review, not committed.

6. **Concept IDs are stable.** `C01–C08` and `S01–S39` are referenced
   from code docstrings and across many doc files. Never renumber them.
   New concepts get the next available ID; new sources append.

## Python environment

- Project is installable editable: `pip install -e '.[dev]'` from the
  repo root (pulls in `pyboy`, `pufferlib`, `torch`, `wandb`, etc.).
- **`pokemonred_puffer.data.events` imports `pyboy` at module level**, so
  importing `pokemonred_puffer` anywhere requires the full install — no
  stdlib-only subset. This is why `scripts/check_docs.py` includes a
  `sys.path.insert(0, REPO_ROOT)` so it works without a separate
  editable install, and also handles the case where the import fails by
  skipping the primitives/dashboard checks.

## Doc tree orientation

Main entry points, in suggested reading order for a new session:

1. [`README.md`](README.md) — top-level project README; links to both
   `rubric_rl` and `goal_rl` docs.
2. [`docs/goal_rl.md`](docs/goal_rl.md) — canonical design doc for the
   goal-setting layer. Architecture, Code Map, Quick Start, How It
   Works, Configuration Reference, Key Concepts (pointers), How to
   Extend, Testing & Verification, Known Limitations, Decision Log
   (summary), Research Notes (pointer).
3. [`docs/goal_rl/`](docs/goal_rl/README.md) — long-form material:
   - `concepts/C01–C08` — per-concept deep dives.
   - `adr/0001–0016` — decision records with context/decision/
     consequences.
   - `research/` — psychology/game-studies/RL literature notes +
     bibliography.
4. [`docs/rubric_rl.md`](docs/rubric_rl.md) — the baseline GRPO path
   that `goal_rl` wraps.

## When in doubt

- **For routine additions** (new primitive, new criterion, tweak
  trigger thresholds, fix a test): follow the "How to Extend" recipes
  in `docs/goal_rl.md`. Update docs + tests alongside code changes.
- **For novel design work**: research first, discuss with the user,
  draft an ADR, then implement.
- **Before committing anything**: ask the user.
- **If drift-check fails**: fix the doc or the code so they agree —
  usually the doc. Never silence the check without understanding why.

## Pick-up pointers for likely next sessions

- "Run a real training experiment" → ADR 0012 + the CLI smoke test
  above + set `goal_rl.runtime.max_llm_revisions` based on budget.
- "Implement a V2 feature" → find its entry in ADR 0016 "Not in V1",
  scope it, open an ADR for the approach, then proceed.
- "Add a new Layer-2 category / primitive / constraint kind" → follow
  the matching recipe under "How to Extend" in `docs/goal_rl.md`.
- "Clean up pre-existing lint in `rubric_rl/grpo.py`" → 5 flagged
  errors, mostly unused imports. Separate concern from goal_rl.
