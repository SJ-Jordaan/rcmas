from __future__ import annotations

import random
from dataclasses import dataclass

from qlearning.engine import Coord, GameState

from .encoding import TabularEncoding
from .qtable import QTable


@dataclass(slots=True)
class QLearningPolicy:
    qtable: QTable
    encoding: TabularEncoding
    epsilon: float
    rng: random.Random

    def select_action(self, state: GameState) -> Coord | None:
        options = list(state.available_sectors())
        if not options:
            return None

        if self.rng.random() < self.epsilon:
            return self.rng.choice(options)

        sk = self.encoding.state_key(state)
        action_keys = [self.encoding.action_key(state, c) for c in options]
        best_ak = self.qtable.best_action(sk, action_keys)
        if best_ak is None:
            return self.rng.choice(options)
        return self.encoding.decode_action(state, best_ak)
