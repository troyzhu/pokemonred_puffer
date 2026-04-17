# Research notes — RL techniques (R-series)

Compact notes on RL literature grounding the goal_rl design. Source
IDs map to [sources.md](sources.md).

## R01 — Options Framework

- Sutton, Precup & Singh 1999: option = (policy, termination,
  initiation). [[S19](sources.md#s19)]
- Enables planning / learning over temporally extended actions.
  [[S19](sources.md#s19)]
- **Role.** Conceptual template for sub-goals as temporally-extended
  reward-shaped behaviour. Not implemented in V1; informs the
  hierarchy mental model.

## R02 — FeUdal Networks

- Vezhnevets et al. 2017 ICML: manager emits latent goals; worker
  follows. [[S20](sources.md#s20)]
- Separate learning signals at each level. [[S20](sources.md#s20)]
- **Role.** Architectural inspiration. Our hierarchy is over *rewards*
  (rubric shaping), not latent goal vectors.

## R03 — HIRO

- Nachum et al. 2018 NeurIPS: data-efficient HRL with **goal
  relabeling** to handle non-stationary lower levels.
  [[S21](sources.md#s21)]
- **Role.** Precedent for handling non-stationarity when sub-rubrics
  change.

## R04 — HER

- Andrychowicz et al. 2017 NeurIPS: hindsight relabeling — treat
  achieved state as the intended goal. [[S22](sources.md#s22)]
- Transforms sparse-reward problems into dense ones.
  [[S22](sources.md#s22)]
- **Role.** Useful pattern if we later want to salvage training
  episodes after a rubric revision (V2).

## R05 — UVFA

- Schaul et al. 2015 ICML: value function conditioned on goal vector
  → single policy can pursue many goals. [[S23](sources.md#s23)]
- **Role.** Relevant if we later condition the policy on Layer-2
  config vector (V2 idea).

## R06 — Prioritized Level Replay

- Jiang, Grefenstette, Rocktäschel, ICML 2021 (PMLR v139): prioritize
  training on levels whose predicted *learning potential* is highest
  (TD error magnitude). [[S24](sources.md#s24)]
- Curriculum emerges automatically. [[S24](sources.md#s24)]
- **Role.** Direct inspiration for
  [C04](../concepts/C04-revision-triggers-and-safety-nets.md) plateau
  detection: low learning-potential = plateau = trigger revision.

## R07 — PAIRED

- Dennis et al. NeurIPS 2020: adversary designs environments at the
  agent's capability frontier. [[S25](sources.md#s25)]
- **Role.** Inspiration for tightening rubric thresholds as the agent
  improves (V2 feature).

## R08 — ICM

- Pathak et al. ICML 2017: intrinsic reward = forward-model prediction
  error. [[S26](sources.md#s26)]
- Drives exploration in sparse-reward environments.
  [[S26](sources.md#s26)]
- **Role.** A concrete implementation of `discovery` Layer-2 goal if
  we want intrinsic-motivation augmentation.

## R09 — RND

- Burda et al. ICLR 2019: intrinsic reward = prediction error to a
  *fixed random network*. [[S27](sources.md#s27)]
- Simpler than ICM; works well on Montezuma's Revenge.
  [[S27](sources.md#s27)]
- **Role.** Alternative `discovery` implementation; recommended over
  ICM because simpler.

## R10 — Go-Explore

- Ecoffet et al. *Nature* 2021: archive of promising states; "first
  return, then explore." [[S28](sources.md#s28)]
- Solved Montezuma's Revenge, Pitfall. [[S28](sources.md#s28)]
- **Role.** Blueprint for state-archive checkpointing at milestones.
  Existing `GRPOStateManager` in `rubric_rl` is already in this
  spirit.

## R11 — Agent57

- Badia et al. ICML 2020: meta-controller that mixes a family of
  exploration/exploitation regimes. [[S29](sources.md#s29)]
- First to exceed human on all 57 Atari games.
  [[S29](sources.md#s29)]
- **Role.** Precedent for a meta-controller over reward
  configurations — roughly what our revision engine does.

## R12 — Reflexion

- Shinn et al. NeurIPS 2023: agents convert scalar/binary reward into
  *verbal* reflections; feed back as context.
  [[S30](sources.md#s30)]
- Learning without weight updates. [[S30](sources.md#s30)]
- **Role.** Template for the LLM's "why did this sub-rubric
  underperform?" step in
  [C03](../concepts/C03-constrained-llm-revision-engine.md).

## R13 — Eureka

- Ma et al. ICLR 2024: LLM generates reward functions as **code**;
  iteratively refined from training feedback via in-context learning.
  [[S31](sources.md#s31)]
- Outperforms human-designed rewards on 83% of 29 benchmarks.
  [[S31](sources.md#s31)]
- **Role.** Closest peer-reviewed precedent for LLM-driven rubric
  revision. We *constrain* their approach — primitives library
  instead of open code — for safety
  ([C03](../concepts/C03-constrained-llm-revision-engine.md)).

## R14 — SayCan

- Ahn et al. CoRL 2022: score actions as P(useful | instruction) ×
  P(succeeds | state); LLM + affordance value function.
  [[S32](sources.md#s32)]
- **Role.** Direct precedent for "LLM proposes, feasibility layer
  filters" in
  [C03](../concepts/C03-constrained-llm-revision-engine.md).

## R15 — ELLM

- Du et al. ICML 2023: LLM-generated goals as exploration bonus.
  [[S33](sources.md#s33)]
- **Role.** Precedent for LLM sub-goal generation. Ours is rubric-
  level, not goal-state-level, but the spirit is the same.

## R16 — AlphaGo / AlphaZero

- Silver et al. *Nature* 2016, 2017: canonical win-rate as terminal
  objective; self-play + MCTS. [[S34](sources.md#s34),
  [S35](sources.md#s35)]
- **Role.** Clean terminal-objective precedent. Ours is messier
  (multi-goal, revising), but the discipline of keeping a *canonical*
  measure (win rate) is the principle we apply in E2
  ([C02](../concepts/C02-frozen-evaluation-protocol.md)).

## R17 — AlphaStar

- Vinyals et al. *Nature* 2019: league training; multi-agent; win
  rate in rated competitive play. [[S36](sources.md#s36)]
- Uses shaped training objectives + league to robustify strategy.
  [[S36](sources.md#s36)]
- **Role.** Precedent for shaped-train + canonical-eval discipline
  ([C02](../concepts/C02-frozen-evaluation-protocol.md)).

## R18 — Constitutional AI

- Bai et al. Anthropic 2022: written "constitution" of principles
  guides model behaviour via RLAIF. [[S37](sources.md#s37)]
- Demonstrates LLMs can self-critique against a specification.
  [[S37](sources.md#s37)]
- **Role.** Grounds the constitution field in
  [C01](../concepts/C01-three-layer-value-schema.md) and the
  LLM-proposes-within-spec pattern in
  [C03](../concepts/C03-constrained-llm-revision-engine.md).

## R19 — Voyager (Tier C)

- Wang et al. 2023 arXiv (NVIDIA + Caltech + UT Austin + UW-Madison +
  Stanford): GPT-4 driven auto-curriculum + skill library in
  Minecraft. [[S38](sources.md#s38)]
- Flagged Tier C — widely cited but not peer-reviewed. Use as
  inspiration only.
- **Role.** Skill-library idea is a V2 goal; V1 focuses on rubric
  revision.

## R20 — DreamerV3 (Tier C)

- Hafner et al., Google DeepMind: world model + latent imagination;
  single hyperparameter set across 150+ tasks.
  [[S39](sources.md#s39)]
- Not published in a peer-reviewed venue as of most recent check;
  arXiv only.
- **Role.** Noted as a future direction if we want a learned world
  model (V2+).
