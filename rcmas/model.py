"""Definitions 1-8: RCMAS model primitives.

Defines the core domain objects for the Region Control Multi-Agent System:
coordinates, territories, game states, and the state-evolution function.

Convention: agents are 0-indexed internally (0..A-1), with -1 denoting
unoccupied sectors.  The paper uses 1-indexed agents with 0=unoccupied;
docstrings note where the mapping applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


UNOWNED: int = -1
"""Sentinel value for an unoccupied sector (paper: 0)."""


# ---------------------------------------------------------------------------
# Def 1: Coordinate
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Coord:
    """A 2-D grid coordinate (x increases rightward, y increases downward)."""

    x: int
    y: int


# ---------------------------------------------------------------------------
# Def 2: 4-connected neighbourhood
# ---------------------------------------------------------------------------

def neighbors4(c: Coord) -> Iterable[Coord]:
    """Yield the four cardinal neighbours of *c* (N, S, E, W)."""
    yield Coord(c.x + 1, c.y)
    yield Coord(c.x - 1, c.y)
    yield Coord(c.x, c.y + 1)
    yield Coord(c.x, c.y - 1)


# ---------------------------------------------------------------------------
# Def 3: Territory
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Territory:
    """An immutable 2-D territory of acquirable sectors.

    Sectors may form any shape; only coordinates present in *sectors* are
    part of the game.  An internal deterministic ordering is maintained for
    use by the SMT encoding and Q-learning state representation.
    """

    sectors: frozenset[Coord]
    _ordered: tuple[Coord, ...]
    _index_by_coord: dict[Coord, int]

    @staticmethod
    def from_ascii(lines: Iterable[str], *, sector: str = ".") -> Territory:
        """Parse an ASCII grid where *sector* marks acquirable cells."""
        rows = [line.rstrip("\n") for line in lines]
        coords: set[Coord] = set()
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == sector:
                    coords.add(Coord(x, y))
        ordered = tuple(sorted(coords, key=lambda c: (c.y, c.x)))
        index_by_coord = {c: i for i, c in enumerate(ordered)}
        return Territory(frozenset(coords), ordered, index_by_coord)

    def ordered_sectors(self) -> tuple[Coord, ...]:
        """Sectors in a deterministic row-major order."""
        return self._ordered

    def index_of(self, coord: Coord) -> int | None:
        """Return the sector index for *coord*, or ``None`` if not in the territory."""
        return self._index_by_coord.get(coord)

    def __len__(self) -> int:
        return len(self.sectors)


# ---------------------------------------------------------------------------
# Def 4-5: State + ownership
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class State:
    """An immutable game state: territory + per-sector ownership + round counter.

    ``owner_by_index[i]`` gives the owner of sector *i* (in the territory's
    ordered-sector indexing): an agent id in ``0..num_agents-1``, or
    ``UNOWNED`` (``-1``) if unclaimed.  Paper convention: agent *k* here
    corresponds to agent *k+1* in the paper; ``UNOWNED`` corresponds to 0.
    """

    territory: Territory
    num_agents: int
    owner_by_index: tuple[int, ...]
    round_index: int = 0

    @staticmethod
    def initial(territory: Territory, num_agents: int) -> State:
        """Create the initial state with all sectors unowned (Def 4)."""
        if num_agents <= 0:
            raise ValueError("num_agents must be >= 1")
        return State(
            territory=territory,
            num_agents=num_agents,
            owner_by_index=(UNOWNED,) * len(territory),
            round_index=0,
        )

    @property
    def ordered_sectors(self) -> tuple[Coord, ...]:
        return self.territory.ordered_sectors()

    def owner_of(self, coord: Coord) -> int | None:
        """Return the owner of *coord* (agent id), or ``None`` if unowned or not in territory."""
        idx = self.territory.index_of(coord)
        if idx is None:
            return None
        owner = self.owner_by_index[idx]
        return None if owner == UNOWNED else owner

    def available_actions(self) -> tuple[Coord, ...]:
        """Return the set of unowned (claimable) sectors in deterministic order (Def 5)."""
        sectors = self.ordered_sectors
        return tuple(sectors[i] for i, o in enumerate(self.owner_by_index) if o == UNOWNED)

    def is_terminal(self) -> bool:
        """True when every sector has been claimed."""
        return all(o != UNOWNED for o in self.owner_by_index)


# ---------------------------------------------------------------------------
# Def 6: State evolution (step)
# ---------------------------------------------------------------------------

class CollisionError(Exception):
    """Raised when two or more agents attempt to claim the same sector."""


def evolve(state: State, actions: dict[int, Coord | None]) -> State:
    """Apply one round of concurrent actions to *state* and return the successor (Def 6).

    Each agent may claim one unowned sector or pass (``None``).  If two or
    more agents target the same sector, a :class:`CollisionError` is raised.

    Returns the new state with updated ownership and an incremented round
    counter.
    """
    if state.is_terminal():
        return state

    for agent_id in actions:
        if agent_id < 0 or agent_id >= state.num_agents:
            raise ValueError(f"invalid agent_id: {agent_id}")

    chosen: dict[Coord, list[int]] = {}
    for agent_id in range(state.num_agents):
        coord = actions.get(agent_id)
        if coord is None:
            continue
        idx = state.territory.index_of(coord)
        if idx is None:
            raise ValueError(f"action not in territory: {coord}")
        if state.owner_by_index[idx] != UNOWNED:
            raise ValueError(f"action targets owned sector: {coord}")
        chosen.setdefault(coord, []).append(agent_id)

    collisions = [coord for coord, ids in chosen.items() if len(ids) >= 2]
    if collisions:
        raise CollisionError(f"collision at {collisions}")

    owners = list(state.owner_by_index)
    for coord, ids in chosen.items():
        idx = state.territory.index_of(coord)
        assert idx is not None
        owners[idx] = ids[0]

    return State(
        territory=state.territory,
        num_agents=state.num_agents,
        owner_by_index=tuple(owners),
        round_index=state.round_index + 1,
    )


# ---------------------------------------------------------------------------
# Def 7: Adjacency + connected components
# ---------------------------------------------------------------------------

def adjacent_owned(state: State, agent_id: int) -> set[tuple[int, int]]:
    """Return pairs (i, j) of sector indices that are 4-adjacent and both owned by *agent_id*."""
    sectors = state.ordered_sectors
    index_by_coord = state.territory._index_by_coord
    pairs: set[tuple[int, int]] = set()
    for i, c in enumerate(sectors):
        if state.owner_by_index[i] != agent_id:
            continue
        for nb in neighbors4(c):
            j = index_by_coord.get(nb)
            if j is not None and state.owner_by_index[j] == agent_id:
                lo, hi = (i, j) if i < j else (j, i)
                pairs.add((lo, hi))
    return pairs


def connected_components(state: State, agent_id: int) -> list[set[Coord]]:
    """Return the list of 4-connected components for *agent_id*'s owned sectors (Def 7)."""
    owned = {c for c in state.ordered_sectors if state.owner_of(c) == agent_id}
    seen: set[Coord] = set()
    components: list[set[Coord]] = []

    for start in owned:
        if start in seen:
            continue
        component: set[Coord] = set()
        stack = [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            component.add(cur)
            for nb in neighbors4(cur):
                if nb in owned and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(component)

    return components


# ---------------------------------------------------------------------------
# Def 8: Largest region size (payoff primitive)
# ---------------------------------------------------------------------------

def largest_region_size(state: State, agent_id: int) -> int:
    """Size of the largest 4-connected region owned by *agent_id* (Def 8)."""
    comps = connected_components(state, agent_id)
    return max((len(c) for c in comps), default=0)


def scores(state: State) -> tuple[int, ...]:
    """Convenience: largest-region-size for each agent."""
    return tuple(largest_region_size(state, i) for i in range(state.num_agents))
