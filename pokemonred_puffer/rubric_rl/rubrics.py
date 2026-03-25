from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from omegaconf import DictConfig

from pokemonred_puffer.data.events import REQUIRED_EVENTS
from pokemonred_puffer.data.items import REQUIRED_ITEMS, USEFUL_ITEMS


@dataclass
class GameStateSnapshot:
    """Structured game state extracted from RedGymEnv for rubric evaluation."""

    badges: int = 0
    completed_required_events: int = 0
    total_required_events: int = len(REQUIRED_EVENTS)
    completed_required_items: int = 0
    total_required_items: int = len(REQUIRED_ITEMS)
    completed_useful_items: int = 0
    total_useful_items: int = len(USEFUL_ITEMS)
    hm_count: int = 0
    total_hms: int = 5
    party_count: int = 0
    party_levels: list[int] = field(default_factory=list)
    max_level_sum: int = 0
    seen_pokemon_count: int = 0
    caught_pokemon_count: int = 0
    total_pokemon: int = 151
    unique_moves: int = 0
    exploration_tile_count: float = 0.0
    unique_maps_visited: int = 0
    npcs_talked: int = 0
    hidden_objs_found: int = 0
    signs_read: int = 0
    pokecenter_heals: int = 0
    cut_tiles_used: int = 0
    surf_tiles_used: int = 0
    total_steps: int = 0
    max_steps: int = 1
    blackout_count: int = 0


@dataclass
class RubricResult:
    """Result of scoring a game state against a rubric."""

    rubric_name: str
    criterion_scores: dict[str, float]
    criterion_weights: dict[str, float]
    total_score: float


@dataclass
class CompositeRubricResult:
    """Result of scoring a game state against multiple rubrics."""

    rubric_results: dict[str, RubricResult]
    rubric_weights: dict[str, float]
    total_score: float


@dataclass
class Criterion:
    """A single evaluation criterion within a rubric."""

    name: str
    description: str
    weight: float
    evaluate: Callable[[GameStateSnapshot], float]
    category: str = "general"


@dataclass
class Rubric:
    """A collection of weighted criteria for evaluating game state."""

    name: str
    criteria: list[Criterion]

    def score(self, snapshot: GameStateSnapshot) -> RubricResult:
        scores = {}
        weights = {}
        for c in self.criteria:
            raw = c.evaluate(snapshot)
            scores[c.name] = float(np.clip(raw, 0.0, 1.0))
            weights[c.name] = c.weight

        weight_sum = sum(weights.values())
        if weight_sum > 0:
            total = sum(scores[n] * weights[n] for n in scores) / weight_sum
        else:
            total = 0.0

        return RubricResult(
            rubric_name=self.name,
            criterion_scores=scores,
            criterion_weights=weights,
            total_score=total,
        )


@dataclass
class CompositeRubric:
    """Meta-weighted combination of multiple rubrics."""

    rubrics: list[tuple[float, Rubric]]

    def score(self, snapshot: GameStateSnapshot) -> CompositeRubricResult:
        results = {}
        rubric_weights = {}
        for meta_weight, rubric in self.rubrics:
            result = rubric.score(snapshot)
            results[rubric.name] = result
            rubric_weights[rubric.name] = meta_weight

        weight_sum = sum(rubric_weights.values())
        if weight_sum > 0:
            total = (
                sum(rubric_weights[name] * r.total_score for name, r in results.items())
                / weight_sum
            )
        else:
            total = 0.0

        return CompositeRubricResult(
            rubric_results=results,
            rubric_weights=rubric_weights,
            total_score=total,
        )


# --- Pokemon Red Rubric Definitions ---


def story_progression_rubric() -> Rubric:
    return Rubric(
        name="story_progression",
        criteria=[
            Criterion(
                name="badges",
                description="Gym badges obtained (out of 8)",
                weight=10.0,
                evaluate=lambda s: s.badges / 8.0,
                category="progression",
            ),
            Criterion(
                name="required_events",
                description="Key story events completed",
                weight=7.0,
                evaluate=lambda s: s.completed_required_events / max(s.total_required_events, 1),
                category="progression",
            ),
            Criterion(
                name="required_items",
                description="Key items obtained",
                weight=5.0,
                evaluate=lambda s: s.completed_required_items / max(s.total_required_items, 1),
                category="progression",
            ),
            Criterion(
                name="hm_count",
                description="HMs obtained (out of 5)",
                weight=8.0,
                evaluate=lambda s: s.hm_count / max(s.total_hms, 1),
                category="progression",
            ),
            Criterion(
                name="useful_items",
                description="Useful items obtained",
                weight=2.0,
                evaluate=lambda s: s.completed_useful_items / max(s.total_useful_items, 1),
                category="progression",
            ),
        ],
    )


def exploration_rubric() -> Rubric:
    # Normalize exploration by a reasonable upper bound
    MAX_TILES = 5000.0
    MAX_MAPS = 100.0
    MAX_NPCS = 200.0
    MAX_HIDDEN = 50.0

    return Rubric(
        name="exploration",
        criteria=[
            Criterion(
                name="tiles_explored",
                description="Unique map tiles visited",
                weight=3.0,
                evaluate=lambda s: min(s.exploration_tile_count / MAX_TILES, 1.0),
                category="exploration",
            ),
            Criterion(
                name="maps_visited",
                description="Unique maps/areas entered",
                weight=4.0,
                evaluate=lambda s: min(s.unique_maps_visited / MAX_MAPS, 1.0),
                category="exploration",
            ),
            Criterion(
                name="npcs_talked",
                description="NPCs interacted with",
                weight=1.0,
                evaluate=lambda s: min(s.npcs_talked / MAX_NPCS, 1.0),
                category="exploration",
            ),
            Criterion(
                name="hidden_objects",
                description="Hidden objects found",
                weight=1.5,
                evaluate=lambda s: min(s.hidden_objs_found / MAX_HIDDEN, 1.0),
                category="exploration",
            ),
            Criterion(
                name="signs_read",
                description="Signs and landmarks read",
                weight=0.5,
                evaluate=lambda s: min(s.signs_read / 50.0, 1.0),
                category="exploration",
            ),
            Criterion(
                name="field_move_usage",
                description="Cut and Surf tiles used",
                weight=2.0,
                evaluate=lambda s: min((s.cut_tiles_used + s.surf_tiles_used) / 100.0, 1.0),
                category="exploration",
            ),
        ],
    )


def team_building_rubric() -> Rubric:
    return Rubric(
        name="team_building",
        criteria=[
            Criterion(
                name="caught_pokemon",
                description="Pokemon caught",
                weight=3.0,
                evaluate=lambda s: min(s.caught_pokemon_count / 30.0, 1.0),
                category="team",
            ),
            Criterion(
                name="seen_pokemon",
                description="Pokemon seen",
                weight=1.5,
                evaluate=lambda s: min(s.seen_pokemon_count / 60.0, 1.0),
                category="team",
            ),
            Criterion(
                name="team_level",
                description="Average team level relative to progression",
                weight=4.0,
                evaluate=lambda s: (
                    min(sum(s.party_levels) / max(len(s.party_levels), 1) / 50.0, 1.0)
                    if s.party_levels
                    else 0.0
                ),
                category="team",
            ),
            Criterion(
                name="team_size",
                description="Party size (more pokemon = better coverage)",
                weight=2.0,
                evaluate=lambda s: min(s.party_count / 6.0, 1.0),
                category="team",
            ),
            Criterion(
                name="move_diversity",
                description="Unique moves learned across party",
                weight=1.5,
                evaluate=lambda s: min(s.unique_moves / 20.0, 1.0),
                category="team",
            ),
        ],
    )


def resource_management_rubric() -> Rubric:
    return Rubric(
        name="resource_management",
        criteria=[
            Criterion(
                name="pokecenter_usage",
                description="Uses pokecenter to heal",
                weight=2.0,
                evaluate=lambda s: min(s.pokecenter_heals / 10.0, 1.0),
                category="resources",
            ),
            Criterion(
                name="efficiency",
                description="Steps used efficiently (lower is better)",
                weight=3.0,
                evaluate=lambda s: (
                    1.0
                    - min(s.total_steps / max(s.max_steps, 1), 1.0)
                    + 0.5 * (s.completed_required_events / max(s.total_required_events, 1))
                )
                / 1.5,
                category="resources",
            ),
            Criterion(
                name="survival",
                description="Low blackout count",
                weight=2.0,
                evaluate=lambda s: max(1.0 - s.blackout_count / 10.0, 0.0),
                category="resources",
            ),
        ],
    )


# --- Factory ---

DEFAULT_RUBRIC_WEIGHTS = {
    "story_progression": 3.0,
    "exploration": 1.0,
    "team_building": 0.5,
    "resource_management": 0.3,
}

RUBRIC_FACTORIES = {
    "story_progression": story_progression_rubric,
    "exploration": exploration_rubric,
    "team_building": team_building_rubric,
    "resource_management": resource_management_rubric,
}


def build_composite_rubric(config: DictConfig | dict | None = None) -> CompositeRubric:
    """Build a CompositeRubric from config. Uses defaults if no config provided."""
    if config is None:
        weights = DEFAULT_RUBRIC_WEIGHTS
    elif isinstance(config, DictConfig):
        weights = dict(config.get("rubrics", DEFAULT_RUBRIC_WEIGHTS))
    else:
        weights = config.get("rubrics", DEFAULT_RUBRIC_WEIGHTS)

    rubrics = []
    for name, factory in RUBRIC_FACTORIES.items():
        meta_weight = weights.get(name, 1.0)
        rubrics.append((meta_weight, factory()))

    return CompositeRubric(rubrics=rubrics)
