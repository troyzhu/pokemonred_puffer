import numpy as np
from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.data.events import REQUIRED_EVENTS
from pokemonred_puffer.data.items import HM_ITEMS, REQUIRED_ITEMS, USEFUL_ITEMS
from pokemonred_puffer.data.tilesets import Tilesets
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.rubric_rl.rubrics import (
    CompositeRubric,
    GameStateSnapshot,
    build_composite_rubric,
)


class RubricRewardEnv(RedGymEnv):
    """Environment that uses rubric-based scoring for rewards.

    Works in two modes:
    - PPO mode: get_game_state_reward() returns rubric criterion scores as a dict
      (compatible with existing update_reward() delta mechanism)
    - GRPO mode: extract_snapshot() provides a GameStateSnapshot for episode-level
      rubric evaluation by the GRPO trainer. The snapshot is also emitted in the
      info dict at episode boundaries so the GRPO trainer can retrieve it through
      the vectorized env interface.
    """

    def __init__(self, env_config: DictConfig, reward_config: DictConfig):
        super().__init__(env_config)
        self.reward_config = OmegaConf.to_object(reward_config)
        self.max_event_rew = 0
        self._pokecenter_heal_count = 0

        rubric_config = self.reward_config.get("rubric_config", None)
        self.rubric: CompositeRubric = build_composite_rubric(rubric_config)

    def pokecenter_heal_hook(self, *args, **kwargs):
        super().pokecenter_heal_hook(*args, **kwargs)
        self._pokecenter_heal_count += 1

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # Emit snapshot in info dict at every step where info is non-empty,
        # and always at episode end (done=True), so the GRPO trainer can
        # retrieve the final game state for rubric scoring.
        if done:
            snapshot = self.extract_snapshot()
            # Store as a dataclass instance, NOT a dict, so that
            # pufferlib's unroll_nested_dict won't flatten it.
            info["rubric_snapshot"] = snapshot
            info["rubric_env_id"] = self.env_id
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self._pokecenter_heal_count = 0
        return super().reset(seed=seed, options=options)

    def get_game_state_reward(self) -> dict[str, float]:
        """Return rubric criterion scores as weighted dict (PPO-compatible)."""
        snapshot = self.extract_snapshot()
        result = self.rubric.score(snapshot)

        reward_dict = {}
        for rubric_name, rubric_result in result.rubric_results.items():
            meta_weight = result.rubric_weights[rubric_name]
            for crit_name, crit_score in rubric_result.criterion_scores.items():
                crit_weight = rubric_result.criterion_weights[crit_name]
                reward_dict[f"{rubric_name}/{crit_name}"] = (
                    meta_weight * crit_weight * crit_score
                )

        return reward_dict

    def extract_snapshot(self) -> GameStateSnapshot:
        """Build a GameStateSnapshot from current PyBoy/env state."""
        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        numBagItems = self.read_m("wNumBagItems")
        bag_item_ids = set(self.pyboy.memory[wBagItems : wBagItems + 2 * numBagItems : 2])

        party_count = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_count)]

        completed_required_events = sum(
            int(self.events.get_event(event)) for event in REQUIRED_EVENTS
        )
        completed_required_items = sum(
            int(item.value in bag_item_ids) for item in REQUIRED_ITEMS
        )
        completed_useful_items = sum(
            int(item.value in bag_item_ids) for item in USEFUL_ITEMS
        )
        hm_count = len(HM_ITEMS.intersection(bag_item_ids))

        exploration_tile_count = sum(
            sum(tileset_coords.values())
            for tileset_coords in self.seen_coords.values()
        )

        return GameStateSnapshot(
            badges=self.get_badges(),
            completed_required_events=completed_required_events,
            total_required_events=len(REQUIRED_EVENTS),
            completed_required_items=completed_required_items,
            total_required_items=len(REQUIRED_ITEMS),
            completed_useful_items=completed_useful_items,
            total_useful_items=len(USEFUL_ITEMS),
            hm_count=hm_count,
            total_hms=5,
            party_count=party_count,
            party_levels=party_levels,
            max_level_sum=self.max_level_sum,
            seen_pokemon_count=int(np.sum(self.seen_pokemon)),
            caught_pokemon_count=int(np.sum(self.caught_pokemon)),
            total_pokemon=151,
            unique_moves=int(np.sum(self.obtained_move_ids)),
            exploration_tile_count=exploration_tile_count,
            unique_maps_visited=int(np.sum(self.seen_map_ids)),
            npcs_talked=len(self.seen_npcs),
            hidden_objs_found=len(self.seen_hidden_objs),
            signs_read=len(self.seen_signs),
            pokecenter_heals=self._pokecenter_heal_count,
            cut_tiles_used=len(self.cut_tiles),
            surf_tiles_used=len(self.surf_tiles),
            total_steps=self.step_count,
            max_steps=self.max_steps,
            blackout_count=self.died_count,
        )
