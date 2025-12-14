from __future__ import annotations

import base64
from dataclasses import dataclass

from qlearning.engine import Coord, GameState
from qlearning.engine.state import UNOWNED


@dataclass(frozen=True, slots=True)
class TabularEncoding:
    """Stable encodings for tabular RL on a fixed Territory.

    State encoding is ownership-only (ordered sectors), encoded as URL-safe base64.
    Action encoding is the sector index as a string.
    """

    num_agents: int

    def state_key(self, state: GameState) -> str:
        # Map owners: UNOWNED(-1) -> 0, agent0->1, agent1->2, ...
        raw = bytes((o + 1) if o != UNOWNED else 0 for o in state.owner_by_index)
        return base64.urlsafe_b64encode(raw).decode("ascii")

    def action_key(self, state: GameState, action: Coord) -> str:
        idx = state.territory.index_of(action)
        if idx is None:
            raise ValueError("action not in territory")
        return str(idx)

    def decode_action(self, state: GameState, action_key: str) -> Coord:
        idx = int(action_key)
        sectors = state.ordered_sectors
        return sectors[idx]
