# Research notes — psychology (P-series)

Compact notes on the psychology literature that grounds the goal_rl
design. Source IDs (`S##`) map to [sources.md](sources.md).

## P01 — Self-Determination Theory (SDT)

- Three basic psychological needs: **competence**, **autonomy**,
  relatedness. [[S03](sources.md#s03), [S04](sources.md#s04),
  [S05](sources.md#s05)]
- Intrinsic motivation is sustained when these are met; extrinsic
  pressure can undermine autonomy. [[S05](sources.md#s05)]
- Applied to video games: competence = sense of effective action,
  autonomy = self-directed play; relatedness is weak in single-player.
  [[S03](sources.md#s03), [S04](sources.md#s04)]
- **Design role.** Anchors Layer 1 of
  [C01](../concepts/C01-three-layer-value-schema.md). Relatedness
  dropped for single-player Red.

## P02 — Goal-Setting Theory

- Specific + challenging goals outperform vague "do-your-best"
  intentions, given commitment + feedback. [[S06](sources.md#s06)]
- **Proximal** sub-goals scaffold **distal** goals — more feedback
  events, faster learning, higher self-efficacy.
  [[S06](sources.md#s06)]
- Goal-performance function is roughly linear in goal difficulty up
  to the limit of ability. [[S06](sources.md#s06)]
- **Design role.** Justifies milestone-based proximal goals in
  [C04](../concepts/C04-revision-triggers-and-safety-nets.md). Informs
  that sub-rubrics should be specific, not vague.

## P03 — Construal Level Theory

- Psychological distance (temporal, spatial, social, hypothetical) is
  represented as **abstract** construal; proximal things as
  **concrete** construal. [[S07](sources.md#s07)]
- Abstract construal emphasizes goals/why; concrete construal
  emphasizes actions/how. [[S07](sources.md#s07)]
- **Design role.** Supports the abstract→concrete tiering in
  [C01](../concepts/C01-three-layer-value-schema.md): Layer 1 abstract,
  Layer 2 concrete goals, sub-criteria concrete state signatures.

## P04 — Cybernetic Self-Regulation / TOTE

- **TOTE** loop (Test–Operate–Test–Exit): compare state to reference,
  act, re-test, exit on match. [[S08](sources.md#s08)]
- Carver & Scheier: hierarchical control — higher-level reference
  values set lower-level ones; discrepancy detection drives
  adjustment. [[S09](sources.md#s09)]
- **Design role.** Abstract template for revision triggers in
  [C04](../concepts/C04-revision-triggers-and-safety-nets.md):
  discrepancy between expected and actual E2 trajectory → revise.

## P05 — Implementation Intentions

- "If X then Y" plans bypass deliberation — the cue X automatically
  primes the response Y. [[S10](sources.md#s10)]
- Robustly shown to improve goal attainment in dozens of peer-reviewed
  studies. [[S10](sources.md#s10)]
- **Design role.** Inspiration for Layer-3 constraint format
  (rule-sets as if-then masks) in
  [C01](../concepts/C01-three-layer-value-schema.md).

## P06 — Metacognitive Monitoring

- Meta-level (monitoring) and object-level (performing) loops;
  judgments of learning / feelings of knowing inform control decisions.
  [[S11](sources.md#s11)]
- Humans re-plan when actual progress diverges from predicted.
  [[S11](sources.md#s11)]
- **Design role.** Justifies plateau detection as a revision trigger
  in [C04](../concepts/C04-revision-triggers-and-safety-nets.md) —
  it's metacognitive monitoring translated into a statistic.

## P07 — Flow Theory

- Optimal engagement when challenge ≈ skill; boredom if too easy,
  anxiety if too hard. [[S12](sources.md#s12)]
- Flow requires clear goals, immediate feedback, sense of control.
  [[S12](sources.md#s12)]
- **Design role.** Argues for difficulty scaling with agent competence
  (V2 feature — not in V1).

## P08 — Self-Efficacy Theory

- Self-efficacy = belief in one's capability to succeed; predicts
  persistence and goal ambition. [[S13](sources.md#s13)]
- Mastery experiences are the strongest efficacy source.
  [[S13](sources.md#s13)]
- **Design role.** Motivates treating recent success rate as an input
  signal to the revision engine when proposing new goal difficulty
  (V2 direction).

## P09 — Curiosity and Information Gap

- **Berlyne:** collative variables (novelty, complexity, uncertainty)
  drive exploration. [[S14](sources.md#s14)]
- **Loewenstein:** curiosity arises from an *information gap* between
  current and desired knowledge state; small gaps especially
  motivating. [[S15](sources.md#s15)]
- **Design role.** Grounds `discovery` Layer-2 goal in
  [C01](../concepts/C01-three-layer-value-schema.md) and aligns with
  [R08 ICM](rl-techniques.md#r08-icm),
  [R09 RND](rl-techniques.md#r09-rnd).

## P10 — Rokeach Values

- Distinction: **terminal values** (end-states to strive for) vs
  **instrumental values** (preferred modes of conduct).
  [[S16](sources.md#s16)]
- Widely adopted in empirical social psych. [[S16](sources.md#s16)]
- **Design role.** Underpins the split in
  [C01](../concepts/C01-three-layer-value-schema.md) between Layer-2
  (terminal = goals) and Layer-3 (instrumental = constraints).
