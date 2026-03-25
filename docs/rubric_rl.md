# Rubric RL for Pokemon Red

## Overview

This document describes our adaptation of Rubric-based Reinforcement Learning (Rubric RL) to train Pokemon Red agents. The original Rubric RL method ([Wolfe, 2025](https://cameronrwolfe.substack.com/p/rubric-rl)) was designed for aligning large language models on subjective, non-verifiable tasks. We adapt its core ideas — structured rubric evaluation and Group Relative Policy Optimization (GRPO) — to the game RL setting.

### Motivation

The baseline Pokemon Red RL system uses PPO with **hand-crafted, manually-weighted reward components** (e.g., `badges: 1.94`, `exploration: 0.029`, `required_event: 7.13`). This approach has several limitations:

1. **Opaque tuning**: Reward weights are found via hyperparameter search (CARBS), producing precise but uninterpretable values like `0.02902716636657715`.
2. **Flat structure**: All reward components are summed into a single scalar with no semantic grouping.
3. **Tight coupling**: Adding or removing a reward signal requires re-tuning all other weights.

Rubric RL addresses these problems by organizing rewards into **structured, interpretable rubrics** with hierarchical weighting, and replacing PPO with GRPO which uses **group-relative advantage normalization** instead of a learned value function.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │      Config (YAML)       │
                    │  rubric weights, GRPO    │
                    │  hyperparameters         │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌────▼─────┐ ┌──────▼───────┐
    │  CompositeRubric│ │GRPOConfig│ │ GRPOState    │
    │  ├─ story_prog  │ │group_size│ │ Manager      │
    │  ├─ exploration  │ │clip_coef │ │ (save states)│
    │  ├─ team_build   │ │kl_coef   │ └──────────────┘
    │  └─ resource_mgmt│ └──────────┘
    └─────────┬──────┘
              │
    ┌─────────▼──────────────────────────────┐
    │              CleanGRPO Trainer          │
    │                                        │
    │  1. collect_episodes()                 │
    │     └─ run vecenv until all envs done  │
    │     └─ store obs/actions/logprobs      │
    │     └─ capture GameStateSnapshots      │
    │                                        │
    │  2. evaluate_episodes_with_rubric()    │
    │     └─ score each snapshot             │
    │     └─ → reward matrix (groups × G)    │
    │                                        │
    │  3. compute_grpo_advantages()          │
    │     └─ A_i = (r_i - μ_group) / σ_group│
    │                                        │
    │  4. train()                            │
    │     └─ clipped policy gradient         │
    │     └─ KL penalty                      │
    │     └─ entropy bonus                   │
    │     └─ NO value loss                   │
    └────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Rubrics

A **rubric** is a collection of weighted **criteria**, each of which evaluates one aspect of game performance on a `[0, 1]` scale. Rubrics are grouped semantically and combined with meta-weights.

#### Rubric Hierarchy

```
CompositeRubric
├── story_progression (weight: 3.0)
│   ├── badges          (w=10.0)  →  badges_obtained / 8
│   ├── required_events (w=7.0)   →  completed_events / total_events
│   ├── required_items  (w=5.0)   →  obtained_items / total_items
│   ├── hm_count        (w=8.0)   →  hms_obtained / 5
│   └── useful_items    (w=2.0)   →  useful_items / total_useful
│
├── exploration (weight: 1.0)
│   ├── tiles_explored    (w=3.0) →  tiles / 5000
│   ├── maps_visited      (w=4.0) →  maps / 100
│   ├── npcs_talked       (w=1.0) →  npcs / 200
│   ├── hidden_objects    (w=1.5) →  found / 50
│   ├── signs_read        (w=0.5) →  signs / 50
│   └── field_move_usage  (w=2.0) →  (cut + surf tiles) / 100
│
├── team_building (weight: 0.5)
│   ├── caught_pokemon  (w=3.0)   →  caught / 30
│   ├── seen_pokemon    (w=1.5)   →  seen / 60
│   ├── team_level      (w=4.0)   →  avg_level / 50
│   ├── team_size       (w=2.0)   →  party_size / 6
│   └── move_diversity  (w=1.5)   →  unique_moves / 20
│
└── resource_management (weight: 0.3)
    ├── pokecenter_usage (w=2.0)  →  heals / 10
    ├── efficiency       (w=3.0)  →  f(steps_used, events_completed)
    └── survival         (w=2.0)  →  max(1 - deaths/10, 0)
```

**Scoring**: Each criterion returns a value in `[0, 1]`. A rubric's total score is the weighted average of its criteria. The composite score is the weighted average of all rubrics.

#### Why Rubrics Over Raw Reward Weights

| Property | Baseline Rewards | Rubric Rewards |
|----------|-----------------|----------------|
| Interpretability | `exploration: 0.02902` | `tiles_explored: 40% of max` |
| Structure | Flat dict of ~30 components | 4 rubrics × 3-6 criteria each |
| Normalization | Raw sums (scale varies) | All criteria on `[0, 1]` |
| Adding criteria | Requires re-tuning everything | Add to rubric, set weight |
| Debugging | Which weight is wrong? | Which rubric/criterion is off? |

### 2. GameStateSnapshot

A `GameStateSnapshot` is a structured dataclass that captures the complete game state relevant to rubric evaluation at a point in time. It is extracted from the PyBoy emulator state by `RubricRewardEnv.extract_snapshot()`.

```python
@dataclass
class GameStateSnapshot:
    badges: int                    # 0-8
    completed_required_events: int # story events cleared
    completed_required_items: int  # key items obtained
    hm_count: int                  # HMs in bag
    party_count: int               # Pokemon in party
    party_levels: list[int]        # levels of each party member
    seen_pokemon_count: int        # Pokedex seen
    caught_pokemon_count: int      # Pokedex caught
    unique_moves: int              # distinct moves across party
    exploration_tile_count: float  # sum of explored tile values
    unique_maps_visited: int       # distinct map IDs entered
    npcs_talked: int               # NPCs interacted with
    hidden_objs_found: int         # hidden items discovered
    pokecenter_heals: int          # times healed at Pokecenter
    cut_tiles_used: int            # tiles Cut was used on
    surf_tiles_used: int           # tiles Surf was used on
    total_steps: int               # steps taken this episode
    blackout_count: int            # times party fainted
    # ... and more
```

Snapshots are emitted in the environment's `info` dict at episode boundaries (`done=True`), allowing the GRPO trainer to retrieve them through pufferlib's vectorized interface.

### 3. GRPO (Group Relative Policy Optimization)

GRPO is the core algorithmic change from the baseline PPO approach. It was introduced in [Shao et al., 2024](https://arxiv.org/abs/2402.03300) for LLM alignment and is the optimizer used in DeepSeek-R1.

#### Key Differences from PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Value network | Yes (critic) | **No** |
| Advantage estimation | GAE (temporal difference) | **Group-relative normalization** |
| Reward granularity | Per-step | **Per-episode** |
| Baseline | Learned value function | **Group mean reward** |
| Data collection | Continuous stepping | **Complete episodes** |

#### Algorithm

Given `N` environments divided into groups of size `G`:

1. **Collect**: Run all environments to episode completion, recording observations, actions, and log-probabilities.

2. **Evaluate**: Score each completed episode with the composite rubric:
   ```
   r_i = CompositeRubric.score(snapshot_i)    # scalar in [0, 1]
   ```

3. **Normalize**: Within each group `g`, compute group-relative advantages:
   ```
   A_i = (r_i - mean(r_g)) / std(r_g)
   ```
   Every step in episode `i` receives advantage `A_i`.

4. **Train**: Clipped policy gradient (like PPO, but without value loss):
   ```
   ratio = exp(log_prob_new - log_prob_old)
   L_clip = max(-A * ratio, -A * clip(ratio, 1-eps, 1+eps))
   L_kl   = beta * mean(log_prob_new - log_prob_old)
   L_ent  = -alpha * entropy
   Loss   = L_clip + L_kl + L_ent       # no value loss term
   ```

#### Why GRPO for Games

- **No value function needed**: Removes the need to train an accurate critic, which is difficult in complex game environments with sparse rewards.
- **Episode-level evaluation**: Natural for games where success is measured at the end (badges obtained, areas explored) rather than per-step.
- **Relative comparison**: "Did this playthrough explore more than others from a similar starting point?" is a more stable learning signal than absolute reward values.

---

## File Structure

```
pokemonred_puffer/
├── rubric_rl/
│   ├── __init__.py
│   ├── rubrics.py           # Criterion, Rubric, CompositeRubric, GameStateSnapshot
│   │                        # Pokemon-specific rubric definitions
│   │                        # build_composite_rubric() factory
│   ├── grpo.py              # CleanGRPO trainer (parallel to cleanrl_puffer.py)
│   │                        # GRPOConfig, GRPOStateManager, GRPOLosses
│   ├── evaluator.py         # RubricEvaluator (rule-based + optional LLM judge)
│   └── state_summarizer.py  # GameStateSnapshot → text (for LLM judge)
├── rewards/
│   ├── baseline.py          # (existing PPO reward classes, unchanged)
│   └── rubric_reward.py     # RubricRewardEnv: RedGymEnv subclass
│                            # Emits snapshots in info dict at episode end
│                            # PPO-compatible via get_game_state_reward()
├── train.py                 # Added train_grpo command
└── config.yaml              # Added rubric_rl section
```

---

## Usage

### Training with GRPO

```bash
# Basic GRPO training
python -m pokemonred_puffer.train train-grpo --rom-path red.gb

# With wandb tracking
python -m pokemonred_puffer.train train-grpo --rom-path red.gb --track

# With debug settings (serial, 1 env)
python -m pokemonred_puffer.train train-grpo --rom-path red.gb --debug

# Customize vectorization
python -m pokemonred_puffer.train train-grpo --rom-path red.gb \
    --vectorization serial
```

### Training with PPO + Rubric Rewards

The rubric reward system also works as a drop-in replacement for the baseline rewards with the existing PPO trainer:

```bash
python -m pokemonred_puffer.train train \
    --reward-name rubric_reward.RubricRewardEnv \
    --rom-path red.gb
```

### Recommended Configuration for GRPO

GRPO collects complete episodes before training, so it requires different scaling than PPO. In `config.yaml`:

```yaml
train:
  # Use fewer envs for GRPO (each runs a full episode)
  num_envs: 16
  num_workers: 4
  env_batch_size: 4
  # Buffer must hold all episode steps: num_envs × max_steps
  batch_size: 320000  # 16 envs × 20000 max_steps

rubric_rl:
  grpo:
    group_size: 4      # 16 envs / 4 = 4 groups
    clip_coef: 0.2
    kl_coef: 0.01
    update_epochs: 3
    minibatch_size: 2048
    learning_rate: 0.0002
    ent_coef: 0.01
```

Key constraint: `num_envs` must be divisible by `group_size`.

---

## Configuration Reference

### `rubric_rl.grpo`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 8 | Number of environments per GRPO group. Rewards are normalized within groups. |
| `clip_coef` | 0.2 | PPO-style clipping coefficient for the importance ratio. |
| `kl_coef` | 0.01 | KL penalty coefficient against collection-time log-probs. |
| `update_epochs` | 3 | Number of passes over collected data per GRPO epoch. |
| `minibatch_size` | 2048 | Number of steps per training minibatch. |
| `learning_rate` | 0.0002 | Adam optimizer learning rate. |
| `max_grad_norm` | 0.5 | Gradient clipping norm. |
| `ent_coef` | 0.01 | Entropy bonus coefficient (encourages exploration). |

### `rubric_rl.rubrics`

Meta-weights for combining rubrics. Higher weight = more influence on the composite score.

| Rubric | Default Weight | Focus |
|--------|---------------|-------|
| `story_progression` | 3.0 | Badges, key events, HMs, items |
| `exploration` | 1.0 | Map coverage, NPCs, hidden objects |
| `team_building` | 0.5 | Pokemon caught, levels, move diversity |
| `resource_management` | 0.3 | Healing, efficiency, survival |

### `rubric_rl.llm_judge`

Optional LLM-based evaluation (disabled by default).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | false | Enable async LLM evaluation. |
| `provider` | anthropic | LLM provider (`anthropic` or `openai`). |
| `model` | claude-haiku-4-5-20251001 | Model to use for evaluation. |
| `eval_frequency` | 100 | Evaluate every N episodes. |
| `blend_weight` | 0.0 | Weight for blending LLM score with rule-based score. |

---

## How It Works: Step by Step

### Episode Collection

```
For each GRPO epoch:
  1. Reset all episode tracking (snapshots, boundaries)
  2. Loop until all N environments complete one episode:
     a. vecenv.recv() → observations, dones, infos
     b. policy.forward(obs) → actions, log_probs
     c. Store (obs, action, log_prob, env_id) in flat buffer
     d. If env done: record episode boundary, extract snapshot from info
     e. vecenv.send(actions)
  3. Result: flat buffer of all steps + per-env snapshots
```

### Rubric Evaluation

```
For each environment i:
  snapshot_i = env's GameStateSnapshot at episode end
  result_i = CompositeRubric.score(snapshot_i)
  reward_i = result_i.total_score   # single scalar in [0, 1]

Reshape into: rewards[num_groups, group_size]
```

### Advantage Computation

```
For each group g:
  μ_g = mean(rewards[g, :])
  σ_g = max(std(rewards[g, :]), 1e-8)
  advantages[g, :] = (rewards[g, :] - μ_g) / σ_g
```

Example with `group_size=4`:
```
Group rewards: [0.15, 0.42, 0.28, 0.35]
Group mean:    0.30
Group std:     0.10
Advantages:    [-1.50, 1.20, -0.20, 0.50]
```

Episodes with above-average rubric scores get positive advantage; below-average get negative. This teaches the policy: "among these attempts, what worked better?"

### Policy Update

```
For each training epoch:
  Shuffle all steps randomly
  For each minibatch:
    Forward pass → new_log_probs, entropy
    ratio = exp(new_log_probs - old_log_probs)
    pg_loss = max(-adv * ratio, -adv * clip(ratio, 1-ε, 1+ε))
    kl_loss = β * mean(new_log_probs - old_log_probs)
    loss = pg_loss - α * entropy + kl_loss
    Backprop and optimizer step
```

---

## Comparison with Baseline

### Baseline PPO Approach
- **Reward**: 30+ hand-tuned weighted components summed per step
- **Advantage**: GAE with learned value function
- **Training**: Continuous stepping, batch_size steps per update
- **Strengths**: Dense per-step signal, proven at scale
- **Weaknesses**: Reward weight sensitivity, opaque tuning

### Rubric RL / GRPO Approach
- **Reward**: Structured rubric evaluation at episode end
- **Advantage**: Group-relative normalization (no value function)
- **Training**: Complete episodes collected, then trained
- **Strengths**: Interpretable criteria, no critic to train, natural for milestone-based evaluation
- **Weaknesses**: Sparser signal (episode-level), requires complete episodes (higher memory), noisier gradients with LSTM

### Hybrid Approach

The `RubricRewardEnv` supports both modes. You can use rubric-based rewards with PPO for dense, per-step signal using the same interpretable criteria:

```bash
# PPO with rubric-structured rewards (best of both worlds)
python -m pokemonred_puffer.train train \
    --reward-name rubric_reward.RubricRewardEnv
```

This gives you the interpretability of rubrics with the training efficiency of PPO.

---

## Extending the Rubrics

### Adding a New Criterion

Add to an existing rubric factory in `rubric_rl/rubrics.py`:

```python
def exploration_rubric() -> Rubric:
    return Rubric(
        name="exploration",
        criteria=[
            # ... existing criteria ...
            Criterion(
                name="warps_discovered",
                description="Warp points discovered",
                weight=1.5,
                evaluate=lambda s: min(s.warps_found / 80.0, 1.0),
                category="exploration",
            ),
        ],
    )
```

Then add the corresponding field to `GameStateSnapshot` and populate it in `RubricRewardEnv.extract_snapshot()`.

### Adding a New Rubric

1. Define the rubric factory in `rubrics.py`:
   ```python
   def battle_strategy_rubric() -> Rubric:
       return Rubric(
           name="battle_strategy",
           criteria=[
               Criterion(
                   name="type_advantage_usage",
                   description="How often super-effective moves are used",
                   weight=5.0,
                   evaluate=lambda s: min(s.super_effective_count / 50.0, 1.0),
                   category="battle",
               ),
               # ... more criteria
           ],
       )
   ```

2. Register it in `RUBRIC_FACTORIES`:
   ```python
   RUBRIC_FACTORIES = {
       "story_progression": story_progression_rubric,
       "exploration": exploration_rubric,
       "team_building": team_building_rubric,
       "resource_management": resource_management_rubric,
       "battle_strategy": battle_strategy_rubric,  # new
   }
   ```

3. Add the weight in `config.yaml`:
   ```yaml
   rubric_rl:
     rubrics:
       battle_strategy: 2.0
   ```

### Enabling the LLM Judge

The optional LLM judge provides a richer evaluation signal by having an LLM evaluate a text summary of the game session:

```yaml
rubric_rl:
  llm_judge:
    enabled: true
    provider: anthropic
    model: claude-haiku-4-5-20251001
    blend_weight: 0.3  # 30% LLM score, 70% rule-based
```

The LLM receives a structured text summary and the rubric criteria, then returns a single `[0, 1]` score. This runs asynchronously and does not block training.

---

## Known Limitations

1. **LSTM during training**: The GRPO training phase processes shuffled minibatches without LSTM state continuity. Collection uses full LSTM context, but training forward passes are stateless. This produces valid but noisier gradients compared to PPO's BPTT-aware training. For best results, consider using a non-recurrent policy.

2. **Group starting states**: The original GRPO paper has each group start from the same "prompt" (starting state). Our adaptation groups environments logically but doesn't enforce identical starting states. The group-relative normalization still provides useful relative signal — "among these diverse attempts, which explored more?" — but the signal is noisier than true same-starting-state comparison.

3. **Memory scaling**: Complete episode collection requires buffering all steps until every environment finishes. With the default `max_steps=19816`, 288 environments would need ~5.7M buffer entries. Use fewer environments (16-32) for GRPO, or reduce `max_steps`.

4. **Episode length variance**: If some environments finish much faster than others, the collection phase wastes compute waiting for slow environments. The buffer cap (`batch_size`) acts as a safety valve — environments that haven't finished when the buffer is full get truncated.

---

## References

- Wolfe, C. (2025). [Rubric-Based Reinforcement Learning](https://cameronrwolfe.substack.com/p/rubric-rl). Overview of rubric-based reward design for RL.
- Shao, Z. et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300). Introduces GRPO.
- DeepSeek-AI (2025). [DeepSeek-R1](https://arxiv.org/abs/2501.12948). Uses GRPO for reasoning model alignment.
- Thatguy11325 et al. [pokemonred_puffer](https://github.com/PWhiddy/PokemonRedExperiments). Baseline Pokemon Red RL system.
