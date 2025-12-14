from __future__ import annotations

import random
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from qlearning.engine import Coord, GameState
from qlearning.rl.encoding import TabularEncoding
from qlearning.rl.qtable import QTable


@dataclass(slots=True)
class QTableAgent:
    """Agent that acts greedily (or epsilon-greedily) from a saved Q-table."""

    qtable: QTable
    num_agents: int
    epsilon: float = 0.0
    rng: random.Random = field(default_factory=random.Random)
    deterministic_ties: bool = True

    @staticmethod
    def load_json(
        path: Path,
        *,
        num_agents: int,
        epsilon: float = 0.0,
        seed: int = 0,
        deterministic_ties: bool = True,
    ) -> "QTableAgent":
        return QTableAgent(
            qtable=QTable.load_json(path),
            num_agents=num_agents,
            epsilon=epsilon,
            rng=random.Random(seed),
            deterministic_ties=deterministic_ties,
        )

    def select_action(self, state: GameState, agent_id: int) -> Coord | None:
        if state.num_agents != self.num_agents:
            raise ValueError("QTableAgent num_agents mismatch")

        options = list(state.available_sectors())
        if not options:
            return None

        enc = TabularEncoding(num_agents=self.num_agents)

        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return self.rng.choice(options)

        sk = enc.state_key(state)
        best: Coord | None = None
        best_v: float | None = None
        for c in options:
            ak = enc.action_key(state, c)
            v = self.qtable.get(sk, ak)
            if best is None or v > best_v:  # type: ignore[operator]
                best, best_v = c, v
            elif v == best_v and best is not None and self.deterministic_ties:
                # Stable tie-break: pick smallest coord in reading order.
                if (c.y, c.x) < (best.y, best.x):
                    best = c

        # If everything is 0 and we want stochastic tie-breaks, pick randomly.
        if best_v == 0.0 and not self.deterministic_ties:
            return self.rng.choice(options)
        return best
