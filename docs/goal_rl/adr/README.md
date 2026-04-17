# Architecture Decision Records — goal_rl

Append-only, dated records of design choices. Each ADR captures **what was
decided**, **why**, and **what it implies** — so a future maintainer can see
the shape of the design *and* the reasoning behind it, not just the end state.

Format: one markdown file per decision, numbered with a zero-padded integer
and a short slug. ADRs are never deleted; if a decision is reversed, the old
file gets `Status: superseded by NNNN` and the new ADR says
`Supersedes NNNN`. The Decision Log section in `docs/goal_rl.md` holds the
summary table.

## Index

| # | Title | Date | Status |
|---|-------|------|--------|
| [0001](0001-scope.md) | Project scope: goal-setting layer on top of rubric_rl | 2026-04-17 | accepted |
| [0002](0002-source-quality-tiers.md) | Strict source-quality tiers for literature | 2026-04-17 | accepted |
| [0003](0003-three-layer-schema.md) | Three-layer value schema (L1 / L2 / L3) | 2026-04-17 | accepted |
| [0004](0004-constitution-concept.md) | Constitution = free-text intent | 2026-04-17 | accepted |
| [0005](0005-frozen-evaluation.md) | Three-layer frozen evaluation (E1/E2/E3) | 2026-04-17 | accepted |
| [0006](0006-constrained-llm-revision.md) | Constrained-LLM revision with primitives library | 2026-04-17 | accepted |
| [0007](0007-revision-cadence.md) | Single revision mechanism, three fire conditions | 2026-04-17 | accepted |
| [0008](0008-eval-cadence-adaptive-k.md) | Hybrid eval cadence with adaptive K | 2026-04-17 | accepted |
| [0009](0009-constitution-parsing-pipeline.md) | Constitution: LLM parse + heuristic fallback | 2026-04-17 | accepted |
| [0010](0010-learned-milestones.md) | Milestones learned via deltas + LLM predicates | 2026-04-17 | accepted |
| [0011](0011-starting-trigger-values.md) | Starting values for plateau + adaptive-K knobs | 2026-04-17 | accepted |
| [0012](0012-experiment-design.md) | First experiment: N=3 matched seeds vs baseline PPO | 2026-04-17 | accepted |
| [0013](0013-layer-two-weights-revisable.md) | Layer-2 weights are revisable (bounded) | 2026-04-17 | accepted |
| [0014](0014-layer-two-categories-closed-v1.md) | Layer-2 categories closed in V1, opens in V2 | 2026-04-17 | accepted |
| [0015](0015-baseline.md) | Baseline to beat: original repo PPO | 2026-04-17 | accepted |
| [0016](0016-v1-implementation-landed.md) | V1 implementation landed | 2026-04-17 | accepted |

## Writing a new ADR

1. Find the next available number: `ls docs/goal_rl/adr/ | sort | tail -3`.
2. Create `NNNN-short-slug.md` copying an existing ADR's frontmatter.
3. Fill in: Context (what forced the decision), Decision, Consequences, Related.
4. Add a row to the table above.
5. In `docs/goal_rl.md::Decision Log`, add a one-line summary with a link.
