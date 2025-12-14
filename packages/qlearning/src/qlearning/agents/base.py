from __future__ import annotations

from typing import Protocol

from qlearning.engine import Coord, GameState


class Agent(Protocol):
    def select_action(self, state: GameState, agent_id: int) -> Coord | None:
        """Choose an unowned sector to capture.

        Return None only if no sectors remain (engine will treat it as no-op).
        """

        raise NotImplementedError
