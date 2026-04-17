# `docs/goal_rl/` — design notes subdirectory

The main user-facing document lives at
[`docs/goal_rl.md`](../goal_rl.md). This subdirectory holds the
longer-form material so the main doc stays scannable:

## Folders

- **[adr/](adr/README.md)** — Architecture Decision Records. One file
  per design decision, dated, append-only. Start here if you want the
  history of **why** the system is the way it is.
- **[concepts/](concepts/README.md)** — Concept pages (C01–C08). One
  file per concept, each covering what it does, where it lives in the
  code, and how it connects to other concepts.
- **[research/](research/README.md)** — Literature notes (P/G/R pages)
  and the tiered bibliography.

## Cross-reference conventions

- ADRs: `[ADR NNNN](../adr/NNNN-slug.md)`
- Concepts: `[C0x](../concepts/C0x-slug.md)`
- Research pages: `[P0x](../research/psychology.md#p0x-slug)` (note
  the `#anchor` — research pages group multiple entries per file)
- Sources: `[S##](../research/sources.md#s##)`

## When adding something new

- **New decision** → new ADR under `adr/`. Add a row to
  [`adr/README.md`](adr/README.md) and a one-line entry to the
  Decision Log in [`../goal_rl.md`](../goal_rl.md).
- **New concept** → new file under `concepts/` with the next `C##`
  number. Link it from [`concepts/README.md`](concepts/README.md) and
  the main doc's concept summary.
- **New citation** → add to [`research/sources.md`](research/sources.md)
  with the right tier. Reference it via `[S##]` from wherever it's
  used. Never remove `[S##]` IDs — they're stable.

## Automation

`scripts/check_docs.py` validates that:

- The primitives list in `concepts/C03-*.md` matches
  `goal_rl.primitives.PRIMITIVES`.
- The E1 dashboard in `concepts/C02-*.md` matches
  `goal_rl.evaluator_frozen.DASHBOARD_FIELDS`.
- All `[C##]`, `[S##]`, `[P##]`, etc. cross-references resolve to
  defined entries.
- File-path references (`pokemonred_puffer/...`) exist on disk.

Run it locally before opening a PR; CI runs it automatically.
