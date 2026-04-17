# Codebase reference (B-series)

Pointers to the pre-existing code that goal_rl builds on.

## B01 — `rubric_rl` module

**Files:**

- `rubric_rl/rubrics.py` — `Criterion`, `Rubric`, `CompositeRubric`,
  `GameStateSnapshot`; `build_composite_rubric(config)` factory; four
  default rubrics (`story_progression`, `exploration`,
  `team_building`, `resource_management`).
- `rubric_rl/grpo.py` — `CleanGRPO` trainer: `collect_episodes`,
  `evaluate_episodes_with_rubric`, `compute_grpo_advantages`, `train`,
  `run_epoch`.
- `rubric_rl/evaluator.py` — `RubricEvaluator` (rule-based + optional
  async LLM judge). Scaffolded but not wired into `CleanGRPO`.
- `rubric_rl/state_summarizer.py` — snapshot → human-readable
  markdown.
- `rewards/rubric_reward.py` — `RubricRewardEnv`: emits
  `GameStateSnapshot` in `info` dict at episode end; also
  PPO-compatible via `get_game_state_reward`.

**Config entry point:** `config.yaml::rubric_rl`, parsed in
`pokemonred_puffer/train.py::train_grpo`.

**Documentation:** [`docs/rubric_rl.md`](../../rubric_rl.md).

## B02 — Extension points used by goal_rl

- `rubric_rl.rubrics.build_composite_rubric` — replaced at runtime by
  `goal_rl.goal_manager.GoalManager.get_rubric`.
- `rubric_rl.grpo.CleanGRPO.run_epoch` — overridden in
  `goal_rl.goal_grpo.GoalGRPO` to hook in trigger checks, frozen eval,
  and revision engine after the inherited collect+train path.
- `rewards.rubric_reward.RubricRewardEnv.extract_snapshot` — source of
  the `GameStateSnapshot` fields referenced by primitives. Snapshot
  schema changes flow through here.
- `rubric_rl.state_summarizer.StateSummarizer` — reused by the
  revision engine to build human-readable game-state summaries for the
  LLM prompt.

## B03 — Baseline: original PPO

**Code:** `pokemonred_puffer/rewards/baseline.py` + the existing
`train.py::train` command.

**Claim per repo docs:** already beats the game with hand-tuned reward
weights (CARBS-tuned values).

**Experiment design (V1):**

- `N=3` matched seeds × {baseline PPO, goal-setting}. Matched =
  identical `(env_seed, policy_seed)` pair per row, so RNG-driven
  variance cancels. 6 runs total.
- Identical wall-clock / sample budget per system.
- **Primary metric:** E2 mean across training — the frozen scalar
  ruler.
- **Secondary metric:** E1 dashboard distribution — Pareto picture of
  what's improving (badges, pokédex, maps, blackouts, …).
- **Tertiary:** wall-clock-per-badge + audit-trail inspection.

**Why matched seeds.** Pokémon Red RL has high cross-seed variance.
Single-seed comparisons can mislead (baseline wins/loses by luck).
Matched seeds remove most of that variance source so we see the
algorithm effect.

**Risk.** The baseline has had significant tuning effort; goal-setting
V1 has not. Compare *trajectories of improvement*, not just end-of-run
scores.

See [ADR 0012](../adr/0012-experiment-design.md) for the formal
decision.
