from __future__ import annotations

import random

from qlearning.engine import Coord, GameState


class RandomAgent:
    def __init__(self, *, rng: random.Random | None = None):
        self._rng = rng or random.Random()

    def select_action(self, state: GameState, agent_id: int) -> Coord | None:
        options = state.available_sectors()
        if not options:
            return None
        return self._rng.choice(options)
