# ADR 0007 — Single revision mechanism, three fire conditions

- **Date:** 2026-04-17
- **Status:** accepted

## Context

Dual-mode schemes (fast online reweighting + slow deep revision) are
expressive but double the failure modes: two cadences, two cooldowns,
two ways to interact with E2. Prioritise debuggability for V1.

But a single fire condition is fragile: if we only fire on milestones,
a stuck agent never sees revisions; if only on plateaus, we miss the
*interesting* moments when the game's phase shifts.

## Decision

**One revision mechanism, three fire conditions** (any one fires):

1. **Milestone trigger** — the eval was milestone-driven. (Almost
   always revise at milestones — the game phase has shifted.)
2. **Plateau trigger** — E2 hasn't improved by ≥ ε over the last N
   eval windows. Grounded in Nelson & Narens metacognitive monitoring.
3. **Revision ceiling** — no revision has fired in M epochs. Forces at
   least one revision per ceiling even if nothing else fires, so the
   system re-examines itself periodically.

Cooldown W between revisions prevents thrashing while the policy
adapts to a new rubric.

## Consequences

- Stuck-agent case (user's concern) is covered: plateau trigger fires
  even without milestones; revision-ceiling is a hard backstop.
- One audit-trail schema for revisions — "what triggered it" is a
  single enum.
- Cooldown + ceiling + watchdog together bound the LLM-call cost.
- V2 may revisit dual-mode if V1 shows the single mechanism is
  insufficient.

## Related

- [0008](0008-eval-cadence-adaptive-k.md) eval cadence (revision is a
  subset of eval events)
- [0005](0005-frozen-evaluation.md) plateau detection runs on E2
- [0011](0011-starting-trigger-values.md) starting numbers
