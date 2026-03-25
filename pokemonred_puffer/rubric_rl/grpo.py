"""GRPO (Group Relative Policy Optimization) trainer for Pokemon Red.

Adapts the Rubric RL / GRPO algorithm from LLM training to game RL:
- Groups of G environments whose rewards are normalized together
- Episode-level rubric scoring
- Group-relative advantage normalization (no value network)
- Clipped policy gradient with KL penalty

Design Notes:
- Observations are stored in a pre-allocated flat buffer (like Experience in
  cleanrl_puffer.py) to avoid OOM from list accumulation.
- Environments are grouped logically (env 0..G-1 = group 0, etc.) for
  advantage normalization. In the full GRPO paper, each group shares the same
  starting state (prompt). Here, we approximate this: environments within a
  group get diverse starting states from the state pool, but their rewards are
  still normalized within the group to provide relative signal. For strict
  same-starting-state semantics, the environment reset mechanism would need
  to be extended.
- LSTM training: during the training phase, we process data in sequential
  chunks per environment to preserve temporal context, similar to the BPTT
  approach in cleanrl_puffer.py.
"""

from __future__ import annotations

import os
import pathlib
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Queue

import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.utils
import pufferlib.vector
import torch
import torch.nn as nn

import wandb
from pokemonred_puffer.cleanrl_puffer import (
    abbreviate,
    count_params,
    seed_everything,
)
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE
from pokemonred_puffer.profile import Profile, Utilization
from pokemonred_puffer.rubric_rl.rubrics import CompositeRubric, GameStateSnapshot


@dataclass
class GRPOConfig:
    """GRPO-specific hyperparameters."""

    group_size: int = 8
    clip_coef: float = 0.2
    kl_coef: float = 0.01
    update_epochs: int = 3
    minibatch_size: int = 2048
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.5
    ent_coef: float = 0.01
    bptt_horizon: int = 16


@dataclass
class GRPOLosses:
    policy_loss: float = 0.0
    entropy: float = 0.0
    kl_penalty: float = 0.0
    approx_kl: float = 0.0
    clipfrac: float = 0.0
    mean_reward: float = 0.0
    mean_within_group_std: float = 0.0


class GRPOStateManager:
    """Manages save-state pool for GRPO group assignments."""

    def __init__(self, state_dir: pathlib.Path, group_size: int, num_groups: int):
        self.state_pool: list[bytes] = []
        self.state_metadata: list[dict] = []
        self.group_size = group_size
        self.num_groups = num_groups
        self._load_initial_states(state_dir)

    def _load_initial_states(self, state_dir: pathlib.Path):
        if not state_dir.exists():
            return
        for state_file in sorted(state_dir.glob("*.state")):
            with open(state_file, "rb") as f:
                self.state_pool.append(f.read())
                self.state_metadata.append({"source": str(state_file), "epoch": 0})

    def assign_states_to_groups(self) -> list[bytes]:
        """Returns num_groups states, each to be shared by group_size envs."""
        if not self.state_pool:
            raise RuntimeError("No save states available in state pool")
        if len(self.state_pool) < self.num_groups:
            return random.choices(self.state_pool, k=self.num_groups)
        return random.sample(self.state_pool, k=self.num_groups)

    def add_state(self, state: bytes, metadata: dict | None = None):
        self.state_pool.append(state)
        self.state_metadata.append(metadata or {})


@dataclass
class CleanGRPO:
    """GRPO training loop for Pokemon Red, parallel to CleanPuffeRL."""

    exp_name: str
    config: object  # train config namespace
    grpo_config: GRPOConfig
    vecenv: pufferlib.vector.Serial | pufferlib.vector.Multiprocessing
    policy: nn.Module
    rubric: CompositeRubric
    state_manager: GRPOStateManager
    env_send_queues: list[Queue]
    env_recv_queues: list[Queue]
    wandb_client: wandb.wandb_sdk.wandb_run.Run | None = None
    profile: Profile = field(default_factory=lambda: Profile())
    losses: GRPOLosses = field(default_factory=lambda: GRPOLosses())
    global_step: int = 0
    epoch: int = 0
    stats: dict = field(default_factory=lambda: {})
    msg: str = ""
    infos: dict = field(default_factory=lambda: defaultdict(list))
    states: dict = field(default_factory=lambda: defaultdict(partial(deque, maxlen=1)))

    def __post_init__(self):
        seed_everything(self.config.seed, self.config.torch_deterministic)

        if self.config.verbose:
            self.utilization = Utilization()

        self.vecenv.async_reset(self.config.seed)

        self.num_envs = self.config.num_envs
        self.num_groups = self.num_envs // self.grpo_config.group_size
        self.group_size = self.grpo_config.group_size

        self.lstm = self.policy.lstm if hasattr(self.policy, "lstm") else None

        self.uncompiled_policy = self.policy
        if self.config.compile:
            self.policy = torch.compile(self.policy, mode=self.config.compile_mode)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.grpo_config.learning_rate, eps=1e-5
        )

        self.last_log_time = time.time()

        # Pre-allocate storage for episode data.
        # GRPO needs complete episodes, so the buffer must hold all steps
        # across all envs until every env finishes one episode.
        # With num_envs=N and max_steps=M, worst case is N*M steps.
        # We store obs on CPU (pinned) to avoid GPU OOM, and transfer
        # to GPU in minibatches during training.
        obs_shape = self.vecenv.single_observation_space.shape
        obs_dtype = self.vecenv.single_observation_space.dtype
        atn_shape = self.vecenv.single_action_space.shape
        torch_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]

        max_episode_steps = getattr(self.config, "max_steps", 20000)
        self._max_buffer_size = self.num_envs * max_episode_steps
        # Cap at a reasonable max to avoid OOM (configurable via batch_size)
        self._max_buffer_size = min(self._max_buffer_size, self.config.batch_size)

        pin = self.config.device == "cuda"
        self._obs_buf = torch.zeros(
            self._max_buffer_size, *obs_shape, dtype=torch_dtype,
            pin_memory=pin,
        )
        self._actions_buf = torch.zeros(
            self._max_buffer_size, *atn_shape, dtype=torch.long,
        )
        self._logprobs_buf = torch.zeros(self._max_buffer_size)
        self._env_ids_buf = np.zeros(self._max_buffer_size, dtype=np.int32)
        self._ptr = 0

        # Per-env episode tracking
        self._env_snapshots: list[GameStateSnapshot | None] = [None] * self.num_envs
        self._env_episode_start: list[int] = [0] * self.num_envs
        self._env_episode_end: list[int] = [0] * self.num_envs

        # LSTM states for collection
        if self.lstm is not None:
            total_agents = self.vecenv.num_agents
            shape = (self.lstm.num_layers, total_agents, self.lstm.hidden_size)
            self._lstm_h = torch.zeros(shape, device=self.config.device)
            self._lstm_c = torch.zeros(shape, device=self.config.device)
        else:
            self._lstm_h = self._lstm_c = None

    @pufferlib.utils.profile
    def collect_episodes(self):
        """Run vecenv until all environments complete one episode.

        Stores transitions in a pre-allocated flat buffer. Tracks episode
        boundaries per environment.
        """
        self._ptr = 0
        for i in range(self.num_envs):
            self._env_snapshots[i] = None
            self._env_episode_start[i] = -1
            self._env_episode_end[i] = -1

        done_envs = set()
        max_steps = self._max_buffer_size

        while len(done_envs) < self.num_envs and self._ptr < max_steps:
            o, r, d, t, info, env_id, mask = self.vecenv.recv()
            env_id_list = env_id.tolist()

            o = torch.as_tensor(o)
            o_device = o.to(self.config.device)

            with torch.no_grad():
                if self._lstm_h is not None:
                    h = self._lstm_h[:, env_id_list]
                    c = self._lstm_c[:, env_id_list]
                    actions, logprob, _, value, (h, c) = self.policy(o_device, (h, c))
                    self._lstm_h[:, env_id_list] = h
                    self._lstm_c[:, env_id_list] = c
                else:
                    actions, logprob, _, value = self.policy(o_device)

            actions_np = actions.cpu().numpy()
            logprob_np = logprob.cpu().numpy()

            if self.num_envs == 1:
                actions_np = np.expand_dims(actions_np, 0)
                logprob_np = np.expand_dims(logprob_np, 0)

            for i, eid in enumerate(env_id_list):
                if not mask[i]:
                    continue
                if eid in done_envs:
                    continue
                if self._ptr >= max_steps:
                    break

                # Record episode start
                if self._env_episode_start[eid] < 0:
                    self._env_episode_start[eid] = self._ptr

                # Store in flat buffer (obs on CPU to save GPU memory)
                ptr = self._ptr
                self._obs_buf[ptr] = o[i]
                self._actions_buf[ptr] = actions[i].cpu()
                self._logprobs_buf[ptr] = logprob[i].cpu()
                self._env_ids_buf[ptr] = eid
                self._ptr += 1
                self.global_step += 1

                if d[i]:
                    self._env_episode_end[eid] = self._ptr
                    done_envs.add(eid)

            # Collect envs that just went done in this batch — we'll use
            # them to assign snapshots via timing-based matching.
            just_done = []
            for i, eid in enumerate(env_id_list):
                if mask[i] and d[i] and eid in done_envs:
                    just_done.append(eid)

            # Extract rubric snapshots and other info from info dicts.
            # Snapshots are matched to envs by timing: a snapshot emitted
            # in the same recv() batch as a done signal belongs to one of
            # the envs that just finished.
            pending_snapshots = []
            for i_info in info:
                if not i_info:
                    continue
                if "rubric_snapshot" in i_info:
                    snapshot = i_info["rubric_snapshot"]
                    if isinstance(snapshot, GameStateSnapshot):
                        pending_snapshots.append(snapshot)

                for k, v in pufferlib.utils.unroll_nested_dict(i_info):
                    if k in ("rubric_snapshot", "rubric_env_id"):
                        continue
                    self.infos[k].append(v)

            # Assign snapshots to just-done envs by order
            for snap, eid in zip(pending_snapshots, just_done):
                self._env_snapshots[eid] = snap

            self.vecenv.send(actions_np)

        # Handle envs that didn't finish (buffer full): mark their end
        for eid in range(self.num_envs):
            if self._env_episode_end[eid] < 0:
                self._env_episode_end[eid] = self._ptr

    def evaluate_episodes_with_rubric(self) -> np.ndarray:
        """Score each episode using the rubric system.

        Returns rewards shaped (num_groups, group_size).
        """
        rewards = np.zeros((self.num_groups, self.group_size), dtype=np.float32)

        for env_idx in range(self.num_envs):
            group_idx = env_idx // self.group_size
            member_idx = env_idx % self.group_size

            snapshot = self._env_snapshots[env_idx]
            if snapshot is not None:
                result = self.rubric.score(snapshot)
                rewards[group_idx, member_idx] = result.total_score
            else:
                # Fallback: minimal snapshot from step count
                ep_len = self._env_episode_end[env_idx] - max(self._env_episode_start[env_idx], 0)
                snapshot = GameStateSnapshot(total_steps=ep_len)
                result = self.rubric.score(snapshot)
                rewards[group_idx, member_idx] = result.total_score

        return rewards

    def compute_grpo_advantages(self, episode_rewards: np.ndarray) -> np.ndarray:
        """Compute group-relative advantages.

        Args:
            episode_rewards: shape (num_groups, group_size)

        Returns:
            advantages: shape (num_groups, group_size) - normalized within each group
        """
        group_means = episode_rewards.mean(axis=1, keepdims=True)
        group_stds = episode_rewards.std(axis=1, keepdims=True)
        group_stds = np.maximum(group_stds, 1e-8)
        advantages = (episode_rewards - group_means) / group_stds
        return advantages

    @pufferlib.utils.profile
    def train(self, advantages: np.ndarray):
        """GRPO training step: clipped policy gradient with episode-level advantages.

        Each step inherits the episode-level advantage of its environment.
        """
        self.losses = GRPOLosses()
        losses = self.losses

        total_steps = self._ptr
        if total_steps == 0:
            return

        # Build per-step advantage tensor from episode-level advantages
        step_advantages = torch.zeros(total_steps, dtype=torch.float32)
        for env_idx in range(self.num_envs):
            group_idx = env_idx // self.group_size
            member_idx = env_idx % self.group_size
            adv_val = advantages[group_idx, member_idx]

            start = self._env_episode_start[env_idx]
            end = self._env_episode_end[env_idx]
            if start >= 0 and end > start:
                # Find all buffer positions for this env
                env_mask = self._env_ids_buf[:total_steps] == env_idx
                step_advantages[torch.from_numpy(env_mask)] = adv_val

        # Slice buffers to actual size
        obs = self._obs_buf[:total_steps]
        actions = self._actions_buf[:total_steps]
        old_logprobs = self._logprobs_buf[:total_steps]

        minibatch_size = min(self.grpo_config.minibatch_size, total_steps)
        num_minibatches = max(total_steps // minibatch_size, 1)

        for epoch_idx in range(self.grpo_config.update_epochs):
            perm = torch.randperm(total_steps)

            epoch_pg_loss = 0.0
            epoch_entropy = 0.0
            epoch_kl = 0.0
            epoch_approx_kl = 0.0
            epoch_clipfrac = 0.0

            for mb in range(num_minibatches):
                start = mb * minibatch_size
                end = min(start + minibatch_size, total_steps)
                mb_idx = perm[start:end]

                mb_obs = obs[mb_idx].to(self.config.device)
                mb_actions = actions[mb_idx].to(self.config.device)
                mb_old_logprobs = old_logprobs[mb_idx].to(self.config.device)
                mb_adv = step_advantages[mb_idx].to(self.config.device)

                # Forward pass — for LSTM, pass state=None.
                # This loses temporal context but avoids needing to sort by
                # (env_id, step) and process in BPTT chunks. The policy still
                # produces valid logprobs, just without recurrent memory.
                # For non-LSTM policies, this is exact.
                if self.lstm is not None:
                    _, new_logprobs, entropy, _, _ = self.policy(
                        mb_obs, state=None, action=mb_actions
                    )
                else:
                    _, new_logprobs, entropy, _ = self.policy(
                        mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape),
                        action=mb_actions,
                    )

                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = (
                        (ratio - 1.0).abs() > self.grpo_config.clip_coef
                    ).float().mean()

                # Normalize advantages within minibatch
                if mb_adv.std() > 1e-8:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio,
                    1 - self.grpo_config.clip_coef,
                    1 + self.grpo_config.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # KL penalty against collection-time policy
                kl_penalty = self.grpo_config.kl_coef * logratio.mean()

                entropy_loss = entropy.mean()

                # Total loss: no value loss in GRPO
                loss = pg_loss - self.grpo_config.ent_coef * entropy_loss + kl_penalty

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.grpo_config.max_grad_norm
                )
                self.optimizer.step()

                if self.config.device == "cuda":
                    torch.cuda.synchronize()

                epoch_pg_loss += pg_loss.item() / num_minibatches
                epoch_entropy += entropy_loss.item() / num_minibatches
                epoch_kl += kl_penalty.item() / num_minibatches
                epoch_approx_kl += approx_kl.item() / num_minibatches
                epoch_clipfrac += clipfrac.item() / num_minibatches

        # Average losses across epochs
        n_epochs = self.grpo_config.update_epochs
        losses.policy_loss = epoch_pg_loss  # last epoch (most relevant)
        losses.entropy = epoch_entropy
        losses.kl_penalty = epoch_kl
        losses.approx_kl = epoch_approx_kl
        losses.clipfrac = epoch_clipfrac

        self.epoch += 1

        # LR annealing
        if self.config.anneal_lr:
            frac = 1.0 - self.global_step / self.config.total_timesteps
            lrnow = frac * self.grpo_config.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # Wandb logging
        if self.wandb_client is not None and time.time() - self.last_log_time > 5.0:
            self.last_log_time = time.time()
            self.wandb_client.log({
                "Overview/agent_steps": self.global_step,
                "Overview/learning_rate": self.optimizer.param_groups[0]["lr"],
                "grpo/policy_loss": losses.policy_loss,
                "grpo/entropy": losses.entropy,
                "grpo/kl_penalty": losses.kl_penalty,
                "grpo/approx_kl": losses.approx_kl,
                "grpo/clipfrac": losses.clipfrac,
                "grpo/mean_reward": losses.mean_reward,
                "grpo/within_group_std": losses.mean_within_group_std,
            })

        if self.epoch % self.config.checkpoint_interval == 0:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

    def run_epoch(self):
        """Run one full GRPO epoch: collect -> evaluate -> train."""
        # Clear infos
        for k in list(self.infos.keys()):
            del self.infos[k]

        # Collect complete episodes into flat buffer
        self.collect_episodes()

        # Score episodes with rubric
        episode_rewards = self.evaluate_episodes_with_rubric()

        # Store reward stats
        self.losses.mean_reward = float(episode_rewards.mean())
        self.losses.mean_within_group_std = float(
            episode_rewards.std(axis=1).mean()
        )

        # Compute group-relative advantages
        advantages = self.compute_grpo_advantages(episode_rewards)

        # Train
        self.train(advantages)

        # Log rubric details
        self._log_rubric_stats()

        if self.config.verbose:
            print(
                f"Epoch {self.epoch} | Steps {self.global_step} | "
                f"Mean reward {self.losses.mean_reward:.4f} | "
                f"PG loss {self.losses.policy_loss:.4f} | "
                f"KL {self.losses.kl_penalty:.4f}"
            )

        return self.stats, self.infos

    def _log_rubric_stats(self):
        """Log per-rubric and per-criterion stats."""
        if self.wandb_client is None:
            return

        all_results = []
        for env_idx in range(self.num_envs):
            snapshot = self._env_snapshots[env_idx]
            if snapshot is not None:
                all_results.append(self.rubric.score(snapshot))

        if not all_results:
            return

        rubric_stats = {}
        for result in all_results:
            for rubric_name, rubric_result in result.rubric_results.items():
                for crit_name, score in rubric_result.criterion_scores.items():
                    key = f"rubric/{rubric_name}/{crit_name}"
                    rubric_stats.setdefault(key, []).append(score)

        log_dict = {k: np.mean(v) for k, v in rubric_stats.items()}
        log_dict["rubric/total_score"] = np.mean(
            [r.total_score for r in all_results]
        )
        self.wandb_client.log(log_dict)

    def done_training(self):
        return self.global_step >= self.config.total_timesteps

    def save_checkpoint(self):
        config = self.config
        path = os.path.join(config.data_dir, config.exp_id)
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f"model_{self.epoch:06d}.pt"
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy, model_path)

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "update": self.epoch,
            "model_name": model_name,
            "exp_id": config.exp_id,
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)
        return model_path

    def close(self):
        self.vecenv.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print("Done GRPO training.")
        self.save_checkpoint()
        self.close()
        print("Run complete")
