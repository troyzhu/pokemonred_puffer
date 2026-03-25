"""Convert GameStateSnapshot to human-readable text for LLM evaluation."""

from __future__ import annotations

from pokemonred_puffer.rubric_rl.rubrics import GameStateSnapshot


class StateSummarizer:
    """Converts game state snapshots to text summaries for LLM judge evaluation."""

    def summarize(self, snapshot: GameStateSnapshot) -> str:
        lines = [
            "=== Pokemon Red Session Summary ===",
            "",
            "## Progression",
            f"- Badges: {snapshot.badges}/8",
            f"- Required events completed: {snapshot.completed_required_events}/{snapshot.total_required_events}",
            f"- Required items obtained: {snapshot.completed_required_items}/{snapshot.total_required_items}",
            f"- Useful items obtained: {snapshot.completed_useful_items}/{snapshot.total_useful_items}",
            f"- HMs obtained: {snapshot.hm_count}/{snapshot.total_hms}",
            "",
            "## Team",
            f"- Party size: {snapshot.party_count}/6",
            f"- Party levels: {snapshot.party_levels if snapshot.party_levels else 'N/A'}",
            f"- Pokemon seen: {snapshot.seen_pokemon_count}/{snapshot.total_pokemon}",
            f"- Pokemon caught: {snapshot.caught_pokemon_count}/{snapshot.total_pokemon}",
            f"- Unique moves learned: {snapshot.unique_moves}",
            "",
            "## Exploration",
            f"- Tiles explored: {snapshot.exploration_tile_count:.0f}",
            f"- Unique maps visited: {snapshot.unique_maps_visited}",
            f"- NPCs talked to: {snapshot.npcs_talked}",
            f"- Hidden objects found: {snapshot.hidden_objs_found}",
            f"- Signs read: {snapshot.signs_read}",
            f"- Cut tiles used: {snapshot.cut_tiles_used}",
            f"- Surf tiles used: {snapshot.surf_tiles_used}",
            "",
            "## Resources",
            f"- Pokecenter heals: {snapshot.pokecenter_heals}",
            f"- Blackouts (deaths): {snapshot.blackout_count}",
            f"- Steps used: {snapshot.total_steps}/{snapshot.max_steps}",
        ]

        return "\n".join(lines)
