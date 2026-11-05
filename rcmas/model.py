"""Chapter 3 of the thesis: RCMAS model primitives.

Defines the core domain objects of a region-control multi-agent system
(Definition 3.1.1): coordinates, territories, control maps (Definition 3.4.1),
the evolution of a state under an action profile (Definition 3.6.1), cohesive
regions (Definition 3.7.1) and rewards (Definition 3.9.2).

Conventions.  Agents are 0-indexed internally (``0..n-1``), and ``UNOWNED``
(``-1``) marks an unoccupied sector; the thesis numbers agents ``1..n`` and
writes ``0`` for an unoccupied sector, so tool agent ``k`` is thesis agent
``k+1``.  Sectors are numbered ``0..|T|-1`` by the enumeration of Remark 3.1.1
(bottom row first, left to right within a row), so tool sector ``i`` is thesis
sector ``i+1``.  A :class:`Coord` keeps the ASCII position of a sector (``y``
is the line number of the grid file, counted from the top), and a sector on
line ``y`` of a grid file with ``h`` lines has thesis coordinates
``(x+1, h-y)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

UNOWNED: int = -1
"""Sentinel value for an unoccupied sector (paper: 0)."""


# ---------------------------------------------------------------------------
# Coordinates (Definition 3.1.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Coord:
    """A 2-D grid coordinate as written in a grid file.

    ``x`` is the column (increasing to the right) and ``y`` the line of the
    grid file (increasing downward).  The thesis draws the grid with ``y``
    increasing upward (Figure 3.1); see the module docstring for the mapping.
    """

    x: int
    y: int


# ---------------------------------------------------------------------------
# Four-connected adjacency (Section 3.2)
# ---------------------------------------------------------------------------


def neighbors4(c: Coord) -> Iterable[Coord]:
    """Yield the four cardinal neighbours of *c* (N, S, E, W)."""
    yield Coord(c.x + 1, c.y)
    yield Coord(c.x - 1, c.y)
    yield Coord(c.x, c.y + 1)
    yield Coord(c.x, c.y - 1)


# ---------------------------------------------------------------------------
# Territory (Definition 3.1.1, Remark 3.1.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Territory:
    """An immutable 2-D territory of acquirable sectors.

    Sectors may form any shape; only coordinates present in *sectors* are
    part of the game.  The sectors are numbered by the enumeration of
    Remark 3.1.1, which the SMT encoding, the canonical completion
    (Definition 4.4.2) and the Q-learning state representation all use.
    """

    sectors: frozenset[Coord]
    _ordered: tuple[Coord, ...]
    _index_by_coord: dict[Coord, int]

    @staticmethod
    def from_ascii(lines: Iterable[str], *, sector: str = ".") -> Territory:
        """Parse an ASCII grid where *sector* marks acquirable cells.

        The first line of the grid is the top row of the territory.  Sectors
        are numbered as in Remark 3.1.1: row by row from the bottom row (the
        last line), and from left to right within a row.
        """
        rows = [line.rstrip("\n") for line in lines]
        coords: set[Coord] = set()
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == sector:
                    coords.add(Coord(x, y))
        ordered = tuple(sorted(coords, key=lambda c: (-c.y, c.x)))
        index_by_coord = {c: i for i, c in enumerate(ordered)}
        return Territory(frozenset(coords), ordered, index_by_coord)

    def ordered_sectors(self) -> tuple[Coord, ...]:
        """Sectors in the enumeration of Remark 3.1.1 (bottom row first)."""
        return self._ordered

    def neighbor_map(self) -> dict[int, frozenset[int]]:
        """Four-connected adjacency between sector indices (Section 3.2)."""
        out: dict[int, frozenset[int]] = {}
        for i, c in enumerate(self._ordered):
            nbs = (self._index_by_coord.get(nb) for nb in neighbors4(c))
            out[i] = frozenset(j for j in nbs if j is not None)
        return out

    def index_of(self, coord: Coord) -> int | None:
        """Return the sector index for *coord*, or ``None`` if not in the territory."""
        return self._index_by_coord.get(coord)

    def __len__(self) -> int:
        return len(self.sectors)


# ---------------------------------------------------------------------------
# Control maps (Definition 3.4.1) and the protocol (Definition 3.5.1)
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
        """Create the initial state ``s_0`` with all sectors unoccupied (Definition 3.4.1)."""
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
        """Return the unoccupied sectors in the enumeration order (Definition 3.5.1)."""
        sectors = self.ordered_sectors
        return tuple(
            sectors[i] for i, o in enumerate(self.owner_by_index) if o == UNOWNED
        )

    def is_terminal(self) -> bool:
        """True when every sector has been claimed."""
        return all(o != UNOWNED for o in self.owner_by_index)


# ---------------------------------------------------------------------------
# Evolution (Definition 3.6.1)
# ---------------------------------------------------------------------------


class CollisionError(Exception):
    """Raised when two or more agents attempt to claim the same sector."""


def evolve(state: State, actions: dict[int, Coord | None]) -> State:
    """Apply one round of concurrent actions to *state* and return the successor.

    Implements case 2 of Definition 3.6.1.  A colliding profile (case 1,
    whose successor is the failure state) raises :class:`CollisionError`.
    For convenience an agent may also pass (``None``), which the thesis
    model does not allow; the synthesisers never pass.

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
# Cohesive regions (Definition 3.7.1)
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
    """Return the cohesive regions of *agent_id* in *state* (Definition 3.7.1)."""
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
# Rewards (Definition 3.9.2)
# ---------------------------------------------------------------------------


def largest_region_size(state: State, agent_id: int) -> int:
    """Reward ``r_a(s)``: size of the largest cohesive region of *agent_id* (Definition 3.9.2)."""
    comps = connected_components(state, agent_id)
    return max((len(c) for c in comps), default=0)


def scores(state: State) -> tuple[int, ...]:
    """Rewards of all agents in *state* (Definition 3.9.2)."""
    return tuple(largest_region_size(state, i) for i in range(state.num_agents))


def largest_component_weight(
    owner_by_index: tuple[int, ...] | list[int],
    agent_id: int,
    neighbors: dict[int, frozenset[int]] | dict[int, list[int]],
    weights: tuple[int, ...] | None = None,
) -> int:
    """Weight of the largest connected component of *agent_id* under *neighbors*.

    With unit weights and four-connected adjacency this is the reward of
    Definition 3.9.2.  With block weights and block adjacency it is the
    abstract reward of Definition 6.1.2(4), where a component's weight is
    the sum of the weights of its blocks.  Sectors are given by index.
    """
    best = 0
    seen: set[int] = set()
    for start, o in enumerate(owner_by_index):
        if o != agent_id or start in seen:
            continue
        seen.add(start)
        stack = [start]
        total = 0
        while stack:
            u = stack.pop()
            total += 1 if weights is None else weights[u]
            for w in neighbors.get(u, ()):
                if w not in seen and owner_by_index[w] == agent_id:
                    seen.add(w)
                    stack.append(w)
        best = max(best, total)
    return best


def rewards_from_owner(
    owner_by_index: tuple[int, ...] | list[int],
    num_agents: int,
    neighbors: dict[int, frozenset[int]] | dict[int, list[int]],
    weights: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Rewards of all agents in a final control vector (Definitions 3.9.2 and 6.1.2(4))."""
    return tuple(
        largest_component_weight(owner_by_index, a, neighbors, weights)
        for a in range(num_agents)
    )
