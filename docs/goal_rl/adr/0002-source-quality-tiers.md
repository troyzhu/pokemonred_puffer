# ADR 0002 — Strict source-quality tiers for literature

- **Date:** 2026-04-17
- **Status:** accepted

## Context

The design draws on psychology, game-studies, and ML literature. A
first sweep surfaced several arXiv preprints from recent months
(2025–2026), Medium posts, and uncurated blog summaries. The user
explicitly audited and rejected these: "peer-reviewed paper,
trustworthy report/blog from respected expert, arXiv only if from
reliable source."

## Decision

Apply a three-tier policy to every citation in goal_rl design docs:

- **Tier A** — peer-reviewed journal/conference papers (preferred).
  Cite author + venue + year.
- **Tier B** — technical reports from reputable labs (DeepMind, OpenAI,
  Anthropic) with named authors.
- **Tier C** — arXiv from a clearly named lab/institution, **flagged as
  Tier C**. Used as design inspiration only, not evidence.
- **Exclude** — Medium, uncurated blogs, unverifiable arXiv IDs,
  secondary summarizers, Wikipedia as primary.

Mark anything uncertain with `[verify]` and prefer under-claiming over
mis-attribution.

## Consequences

- Several weaker citations dropped from the initial sweep.
- Voyager, DreamerV3, and a few other widely-cited arXiv-only papers
  marked as Tier C — usable as design inspiration but never as
  "evidence that X works."
- Every source in `docs/goal_rl/research/sources.md` is grouped by tier.
- Future additions must tag the tier; a stray Medium link is a review bug.

## Related

- [docs/goal_rl/research/sources.md](../research/sources.md) — the
  bibliography with tier groupings.
