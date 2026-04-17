# ADR 0011 — Starting values for plateau + adaptive-K knobs

- **Date:** 2026-04-17
- **Status:** accepted

## Context

ADRs 0007 and 0008 introduce several numerical knobs:
- Plateau detection: `epsilon` (min E2 improvement to count as
  progress), `window` (how many eval windows to require flatness over).
- Adaptive K: `K_min`, `K_max`, `total_epochs` (ramp length).

These all need starting values before the first real training run.
Without data, the values can only be best guesses — they *must* be
treated as knobs to tune empirically.

## Decision

V1 starting defaults (also the config-file defaults):

- `plateau_epsilon = 0.02`
- `plateau_window = 3` eval windows
- `K_min = 2` epochs
- `K_max = 10` epochs
- `total_epochs = 100` (designer overrides based on run length)
- `revision_cooldown_epochs = 3`
- `revision_ceiling_epochs = 30`

Watchdog (ADR 0005):

- `watchdog_window = 3`
- `watchdog_training_rise = 0.05`
- `watchdog_e2_drop = 0.05`

All are config-file knobs under `goal_rl.triggers` and
`goal_rl.evaluator` — tune per run based on observed behaviour.

## Consequences

- Defaults are conservative (frequent evals, tight plateau detection).
- First real run is expected to produce enough audit-log data to tune
  these properly.
- "How to Extend" in the main doc documents the knob-tuning heuristics.

## Related

- [0007](0007-revision-cadence.md) revision cadence
- [0008](0008-eval-cadence-adaptive-k.md) adaptive K
