from __future__ import annotations

from dataclasses import dataclass

from .outcome import GameOutcome, GameStatus
from .territory import Territory
from .types import Coord, neighbors4


UNOWNED = -1


@dataclass(frozen=True, slots=True)
class GameState:
    territory: Territory
    num_agents: int
    owner_by_index: tuple[int, ...]
    round_index: int = 0

    @staticmethod
    def new(territory: Territory, num_agents: int) -> "GameState":
        if num_agents <= 0:
            raise ValueError("num_agents must be >= 1")
        return GameState(
            territory=territory,
            num_agents=num_agents,
            owner_by_index=(UNOWNED,) * len(territory.ordered_sectors()),
            round_index=0,
        )

    @property
    def ordered_sectors(self) -> tuple[Coord, ...]:
        return self.territory.ordered_sectors()

    def owner_of(self, coord: Coord) -> int | None:
        idx = self._index_of(coord)
        if idx is None:
            return None
        owner = self.owner_by_index[idx]
        return None if owner == UNOWNED else owner

    def available_sectors(self) -> tuple[Coord, ...]:
        sectors = self.ordered_sectors
        return tuple(sectors[i] for i, o in enumerate(self.owner_by_index) if o == UNOWNED)

    def is_terminal(self) -> bool:
        return all(o != UNOWNED for o in self.owner_by_index)

    def step(self, actions: dict[int, Coord | None]) -> tuple["GameState", GameOutcome]:
        """Apply all agents' actions concurrently.

        If two or more agents target the same unowned sector, the game ends in defeat.
        If the game fills all sectors, it ends in victory.
        """

        if self.is_terminal():
            return self, GameOutcome(GameStatus.VICTORY, "already terminal")

        # Validate agent ids
        for agent_id in actions.keys():
            if agent_id < 0 or agent_id >= self.num_agents:
                raise ValueError(f"invalid agent_id: {agent_id}")

        # Everyone should act; missing agents mean "no-op".
        chosen: dict[Coord, list[int]] = {}
        for agent_id in range(self.num_agents):
            coord = actions.get(agent_id)
            if coord is None:
                continue
            idx = self._index_of(coord)
            if idx is None:
                raise ValueError(f"action not in territory: {coord}")
            if self.owner_by_index[idx] != UNOWNED:
                raise ValueError(f"action targets owned sector: {coord}")
            chosen.setdefault(coord, []).append(agent_id)

        # Collision => defeat.
        collisions = [coord for coord, ids in chosen.items() if len(ids) >= 2]
        if collisions:
            return self, GameOutcome(GameStatus.DEFEAT, "collision")

        # Apply claims
        owners = list(self.owner_by_index)
        for coord, ids in chosen.items():
            agent_id = ids[0]
            idx = self.territory.index_of(coord)
            if idx is None:
                raise RuntimeError("coord unexpectedly missing from territory")
            owners[idx] = agent_id

        next_state = GameState(
            territory=self.territory,
            num_agents=self.num_agents,
            owner_by_index=tuple(owners),
            round_index=self.round_index + 1,
        )

        if next_state.is_terminal():
            return next_state, GameOutcome(GameStatus.VICTORY, "all sectors acquired")
        return next_state, GameOutcome(GameStatus.ONGOING)

    def largest_region_size(self, agent_id: int) -> int:
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError("invalid agent_id")

        owned = {c for c in self.ordered_sectors if self.owner_of(c) == agent_id}
        seen: set[Coord] = set()
        best = 0

        for start in owned:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            size = 0
            while stack:
                cur = stack.pop()
                size += 1
                for nb in neighbors4(cur):
                    if nb in owned and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            best = max(best, size)

        return best

    def scores(self) -> tuple[int, ...]:
        return tuple(self.largest_region_size(i) for i in range(self.num_agents))

    def _index_of(self, coord: Coord) -> int | None:
        return self.territory.index_of(coord)
