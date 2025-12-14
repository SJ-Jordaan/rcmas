from __future__ import annotations

import random

from qlearning.engine import Coord, GameState, GameStatus


class GreedyLargestRegionAgent:
    """Picks the move that maximizes *immediate* largest-region size."""

    def __init__(self, *, rng: random.Random | None = None):
        self._rng = rng or random.Random()

    def select_action(self, state: GameState, agent_id: int) -> Coord | None:
        options = state.available_sectors()
        if not options:
            return None

        best_val = -1
        best: list[Coord] = []
        for coord in options:
            next_state, outcome = state.step({agent_id: coord})
            if outcome.status == GameStatus.DEFEAT:
                continue
            val = next_state.largest_region_size(agent_id)
            if val > best_val:
                best_val = val
                best = [coord]
            elif val == best_val:
                best.append(coord)

        return self._rng.choice(best) if best else self._rng.choice(options)
