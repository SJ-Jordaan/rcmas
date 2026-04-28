"""Definitions 8-12 (EUMAS): Territory partition and abstract RCMAS.

Provides the data structures and operations for CEGAR-based abstraction
refinement of Nash equilibrium synthesis:

* :class:`Partition` — a partition of concrete sectors into blocks (Def 8)
* :class:`AbstractRCMAS` — the abstract game with weighted blocks (Def 9)
* :func:`lift_strategy` — map abstract solution to concrete strategy (Def 10)
* :func:`compute_deviation_set` — sectors differing between two terminal states (Def 11)
* :func:`refine_partition` — split blocks by deviation set (Def 12)
* :func:`bfs_bisect` — adjacency-aware block bisection for abstract-infeasibility refinement
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .model import UNOWNED, Coord, State, Territory, largest_region_size, neighbors4
from .smt_solve import SmtSolution
from .symmetry import sector_orbits, territory_automorphisms


# ---------------------------------------------------------------------------
# Def 8: Territory Partition
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Partition:
    """A partition of concrete sector indices into blocks.

    Attributes:
        blocks: each block is a frozenset of sector indices
        membership: ``membership[i]`` = block index containing sector *i*
    """

    blocks: tuple[frozenset[int], ...]
    membership: tuple[int, ...]


def discrete_partition(territory: Territory) -> Partition:
    """Create the discrete partition: each sector is its own block."""
    S = len(territory)
    blocks = tuple(frozenset({i}) for i in range(S))
    membership = tuple(range(S))
    return Partition(blocks=blocks, membership=membership)


def orbit_partition(territory: Territory) -> Partition:
    """Create a partition whose blocks are the symmetry orbits of the territory."""
    auts = territory_automorphisms(territory)
    orbits = sector_orbits(territory, auts)
    # Sort orbits by their smallest member for determinism
    orbits_sorted = sorted(orbits, key=lambda orb: min(orb))
    S = len(territory)
    membership = [0] * S
    for block_idx, orb in enumerate(orbits_sorted):
        for i in orb:
            membership[i] = block_idx
    return Partition(
        blocks=tuple(orbits_sorted),
        membership=tuple(membership),
    )


def grid_partition(territory: Territory, tile_w: int, tile_h: int) -> Partition:
    """Partition the territory into rectangular tiles of size *tile_w* × *tile_h*.

    Sectors are grouped by which tile they fall into based on their
    coordinates: sector (x, y) belongs to tile (x // tile_w, y // tile_h).
    Tiles that contain no territory sectors are omitted.
    """
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("tile dimensions must be >= 1")
    sectors = territory.ordered_sectors()
    S = len(sectors)

    # Map each sector to a tile key
    tile_map: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(sectors):
        key = (c.x // tile_w, c.y // tile_h)
        tile_map.setdefault(key, []).append(i)

    # Sort tiles by smallest sector index for determinism
    tiles_sorted = sorted(tile_map.values(), key=lambda lst: min(lst))

    blocks = tuple(frozenset(lst) for lst in tiles_sorted)
    membership = [0] * S
    for block_idx, block in enumerate(blocks):
        for i in block:
            membership[i] = block_idx

    return Partition(blocks=blocks, membership=tuple(membership))


def balanced_partition(territory: Territory, n: int) -> Partition:
    """Partition the territory into *n* connected, equal-sized blocks.

    For rectangular grids this cuts the territory into *n* vertical strips
    (or horizontal strips if the grid is taller than it is wide and *n*
    divides the height).  The result is *n* blocks each of size ⌊S/n⌋ or
    ⌈S/n⌉, giving an abstract horizon of 1 and trivially solvable abstract
    synthesis.

    For non-rectangular territories, falls back to simultaneous BFS (k=*n*)
    which produces approximately balanced connected blocks.
    """
    sectors = territory.ordered_sectors()
    S = len(sectors)
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n >= S:
        return discrete_partition(territory)

    # Determine grid extents
    xs = sorted({c.x for c in sectors})
    ys = sorted({c.y for c in sectors})
    w, h = len(xs), len(ys)

    # Check if territory is a full rectangle
    is_rect = S == w * h

    if is_rect:
        # Try vertical strips first (split along x-axis)
        if w >= n:
            return _strip_partition(sectors, S, n, axis="x", coords=xs)
        # Try horizontal strips (split along y-axis)
        if h >= n:
            return _strip_partition(sectors, S, n, axis="y", coords=ys)

    # Fallback: BFS k-way (connected, approximately balanced)
    return bfs_kway_partition(territory, n)


def _strip_partition(
    sectors: tuple[Coord, ...],
    S: int,
    n: int,
    axis: str,
    coords: list[int],
) -> Partition:
    """Split sectors into *n* contiguous strips along *axis*.

    Strips are created by distributing the unique coordinate values along
    *axis* as evenly as possible among *n* groups.
    """
    num_coords = len(coords)
    # Assign each coordinate value to a strip index
    coord_to_strip: dict[int, int] = {}
    for i, c in enumerate(coords):
        coord_to_strip[c] = i * n // num_coords

    membership = [0] * S
    block_map: dict[int, list[int]] = {}
    for idx, sec in enumerate(sectors):
        strip = coord_to_strip[sec.x if axis == "x" else sec.y]
        membership[idx] = strip
        block_map.setdefault(strip, []).append(idx)

    blocks_sorted = sorted(block_map.values(), key=lambda lst: min(lst))
    blocks = tuple(frozenset(lst) for lst in blocks_sorted)
    # Reassign membership to match sorted block order
    membership = [0] * S
    for block_idx, block in enumerate(blocks):
        for i in block:
            membership[i] = block_idx

    return Partition(blocks=blocks, membership=tuple(membership))


def bfs_kway_partition(territory: Territory, k: int) -> Partition:
    """Partition the territory into *k* blocks via simultaneous BFS from *k* seeds.

    Seeds are chosen by spacing evenly across the row-major sector ordering.
    All seeds expand simultaneously (one hop per round); each sector is
    assigned to the seed that reaches it first (ties broken by seed index).
    This produces a Voronoi-like partition on the grid graph.
    """
    sectors = territory.ordered_sectors()
    S = len(sectors)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k >= S:
        return discrete_partition(territory)

    index_by_coord = {c: i for i, c in enumerate(sectors)}

    # Choose k seeds evenly spaced in row-major order
    seeds = [i * S // k for i in range(k)]

    # Simultaneous BFS
    assignment = [-1] * S
    queues: list[deque[int]] = []
    for seed_idx, seed_sector in enumerate(seeds):
        assignment[seed_sector] = seed_idx
        queues.append(deque([seed_sector]))

    changed = True
    while changed:
        changed = False
        for seed_idx in range(k):
            next_queue: deque[int] = deque()
            while queues[seed_idx]:
                u = queues[seed_idx].popleft()
                for nb in neighbors4(sectors[u]):
                    j = index_by_coord.get(nb)
                    if j is not None and assignment[j] == -1:
                        assignment[j] = seed_idx
                        next_queue.append(j)
                        changed = True
            queues[seed_idx] = next_queue

    # Build blocks from assignment
    block_map: dict[int, list[int]] = {}
    for i, a in enumerate(assignment):
        block_map.setdefault(a, []).append(i)
    blocks_sorted = sorted(block_map.values(), key=lambda lst: min(lst))
    blocks = tuple(frozenset(lst) for lst in blocks_sorted)
    membership = [0] * S
    for block_idx, block in enumerate(blocks):
        for i in block:
            membership[i] = block_idx

    return Partition(blocks=blocks, membership=tuple(membership))


# ---------------------------------------------------------------------------
# Def 9: Abstract RCMAS
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AbstractRCMAS:
    """An abstract RCMAS with weighted blocks (Def 9).

    Attributes:
        partition: the territory partition used
        territory: synthetic Territory with one Coord per block
        weights: ``weights[p]`` = ``|B_p|`` (number of concrete sectors in block *p*)
        neighbors: abstract adjacency (block *p* ~ block *q*)
        horizon: ``len(blocks) // num_agents``
        num_agents: number of agents
        concrete_territory: the original concrete territory
    """

    partition: Partition
    territory: Territory
    weights: tuple[int, ...]
    neighbors: dict[int, list[int]]
    horizon: int
    num_agents: int
    concrete_territory: Territory


def build_abstract_rcmas(
    territory: Territory,
    num_agents: int,
    partition: Partition,
) -> AbstractRCMAS:
    """Construct the abstract RCMAS from a concrete territory and partition (Def 9).

    Creates a synthetic territory with one coordinate per block, derives
    abstract adjacency from the concrete territory, and computes block
    weights as cardinalities.
    """
    k = len(partition.blocks)
    sectors = territory.ordered_sectors()
    concrete_index_by_coord = {c: i for i, c in enumerate(sectors)}

    # Synthetic territory: one Coord(p, 0) per block
    abstract_coords = [Coord(p, 0) for p in range(k)]
    abstract_territory = Territory(
        frozenset(abstract_coords),
        tuple(abstract_coords),
        {c: i for i, c in enumerate(abstract_coords)},
    )

    # Weights = block sizes
    weights = tuple(len(b) for b in partition.blocks)

    # Abstract adjacency: B_p ~ B_q iff ∃ i∈B_p, j∈B_q s.t. i~j in concrete
    abstract_neighbors: dict[int, list[int]] = {p: [] for p in range(k)}
    for p in range(k):
        for q in range(p + 1, k):
            adjacent = False
            for i in partition.blocks[p]:
                if adjacent:
                    break
                coord_i = sectors[i]
                for nb in neighbors4(coord_i):
                    j = concrete_index_by_coord.get(nb)
                    if j is not None and j in partition.blocks[q]:
                        adjacent = True
                        break
            if adjacent:
                abstract_neighbors[p].append(q)
                abstract_neighbors[q].append(p)

    horizon = k // num_agents

    return AbstractRCMAS(
        partition=partition,
        territory=abstract_territory,
        weights=weights,
        neighbors=abstract_neighbors,
        horizon=horizon,
        num_agents=num_agents,
        concrete_territory=territory,
    )


# ---------------------------------------------------------------------------
# Def 10: Strategy Lifting
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LiftedStrategy:
    """A concrete strategy obtained by lifting an abstract solution (Def 10).

    Attributes:
        terminal_owner: per-concrete-sector ownership at the terminal state
        actions_by_round: round-by-round concrete actions per agent
        payoff_by_agent: concrete reward for each agent
    """

    terminal_owner: tuple[int, ...]
    actions_by_round: tuple[tuple[Coord | None, ...], ...]
    payoff_by_agent: tuple[int, ...]


def lift_strategy(
    abstract_rcmas: AbstractRCMAS,
    abstract_solution: SmtSolution,
    concrete_horizon: int | None = None,
) -> LiftedStrategy:
    """Lift an abstract SMT solution to a concrete strategy (Def 10).

    1. Derive concrete terminal ownership from abstract terminal ownership:
       ``owner[i] = abstract_owner[membership[i]]``
    2. Expand abstract round-by-round actions to concrete actions by
       distributing sectors within each block across rounds.
    3. If *concrete_horizon* exceeds the number of rounds produced by the
       abstract assignment, extend the strategy by distributing remaining
       unoccupied sectors round-robin among agents.
    4. Compute concrete rewards on the final terminal state.
    """
    partition = abstract_rcmas.partition
    concrete_territory = abstract_rcmas.concrete_territory
    num_agents = abstract_rcmas.num_agents
    S_concrete = len(concrete_territory)

    assert abstract_solution.final_state is not None
    assert abstract_solution.actions_by_round is not None

    abstract_owner = abstract_solution.final_state.owner_by_index

    # 1. Concrete terminal ownership (from abstract blocks only)
    initial_terminal_owner = list(
        abstract_owner[partition.membership[i]] for i in range(S_concrete)
    )

    # 2. Build concrete actions round by round
    abstract_actions = abstract_solution.actions_by_round
    abstract_horizon = len(abstract_actions)

    agent_blocks: list[list[int]] = [[] for _ in range(num_agents)]
    for t in range(abstract_horizon):
        for a in range(num_agents):
            action_coord = abstract_actions[t][a]
            if action_coord is not None:
                block_idx = abstract_rcmas.territory.index_of(action_coord)
                if block_idx is not None:
                    agent_blocks[a].append(block_idx)

    concrete_sectors = concrete_territory.ordered_sectors()
    agent_concrete_actions: list[list[Coord]] = [[] for _ in range(num_agents)]
    for a in range(num_agents):
        for block_idx in agent_blocks[a]:
            block_sectors = sorted(partition.blocks[block_idx])
            for si in block_sectors:
                agent_concrete_actions[a].append(concrete_sectors[si])

    # 3. Fit to concrete_horizon: each agent makes exactly concrete_horizon
    #    claims (one per round).  Trim agents that have too many, then
    #    distribute remaining unoccupied sectors to agents with room.
    if concrete_horizon is not None:
        target = concrete_horizon

        # Trim over-allocated agents
        for a in range(num_agents):
            if len(agent_concrete_actions[a]) > target:
                agent_concrete_actions[a] = agent_concrete_actions[a][:target]

        # Recompute terminal ownership after trim
        claimed: set[int] = set()
        for a in range(num_agents):
            for coord in agent_concrete_actions[a]:
                idx = concrete_territory.index_of(coord)
                if idx is not None:
                    claimed.add(idx)
                    initial_terminal_owner[idx] = a

        # Mark unclaimed sectors
        for i in range(S_concrete):
            if i not in claimed:
                initial_terminal_owner[i] = UNOWNED

        # Distribute remaining unoccupied sectors round-robin to agents
        # that still have room (fewer than target actions)
        remaining = [i for i in range(S_concrete) if i not in claimed]
        remaining_iter = iter(remaining)
        filled = True
        while filled:
            filled = False
            for a in range(num_agents):
                if len(agent_concrete_actions[a]) >= target:
                    continue
                si = next(remaining_iter, None)
                if si is None:
                    break
                agent_concrete_actions[a].append(concrete_sectors[si])
                initial_terminal_owner[si] = a
                filled = True

        lifted_horizon = target
    else:
        max_actions = max((len(acs) for acs in agent_concrete_actions), default=0)
        lifted_horizon = max_actions if max_actions > 0 else 1

    terminal_owner = tuple(initial_terminal_owner)

    # Build actions_by_round
    actions_by_round: list[tuple[Coord | None, ...]] = []
    for t in range(lifted_horizon):
        step: list[Coord | None] = []
        for a in range(num_agents):
            if t < len(agent_concrete_actions[a]):
                step.append(agent_concrete_actions[a][t])
            else:
                step.append(None)
        actions_by_round.append(tuple(step))

    # 4. Compute concrete rewards on final terminal state
    terminal_state = State(
        territory=concrete_territory,
        num_agents=num_agents,
        owner_by_index=terminal_owner,
        round_index=lifted_horizon,
    )
    payoff_by_agent = tuple(
        largest_region_size(terminal_state, a) for a in range(num_agents)
    )

    return LiftedStrategy(
        terminal_owner=terminal_owner,
        actions_by_round=tuple(actions_by_round),
        payoff_by_agent=payoff_by_agent,
    )


# ---------------------------------------------------------------------------
# Def 11: Deviation Set (Counterexample)
# ---------------------------------------------------------------------------

def compute_deviation_set(
    original_terminal: tuple[int, ...],
    deviated_terminal: tuple[int, ...],
) -> frozenset[int]:
    """Return the set of sector indices whose ownership differs (Def 11)."""
    return frozenset(
        i for i in range(len(original_terminal))
        if original_terminal[i] != deviated_terminal[i]
    )


# ---------------------------------------------------------------------------
# Def 12: Partition Refinement
# ---------------------------------------------------------------------------

def refine_partition(
    partition: Partition,
    deviation_set: frozenset[int],
) -> Partition:
    """Refine *partition* by splitting each block using *deviation_set* (Def 12).

    For each block B_p, if both ``B_p ∩ Δ`` and ``B_p ∖ Δ`` are non-empty,
    replace B_p with the two sub-blocks.  Otherwise keep B_p unchanged.
    """
    new_blocks: list[frozenset[int]] = []
    for block in partition.blocks:
        inside = block & deviation_set
        outside = block - deviation_set
        if inside and outside:
            new_blocks.append(inside)
            new_blocks.append(outside)
        else:
            new_blocks.append(block)

    # Sort blocks by smallest element for determinism
    new_blocks.sort(key=lambda b: min(b))

    S = len(partition.membership)
    new_membership = [0] * S
    for block_idx, block in enumerate(new_blocks):
        for i in block:
            new_membership[i] = block_idx

    return Partition(
        blocks=tuple(new_blocks),
        membership=tuple(new_membership),
    )


# ---------------------------------------------------------------------------
# BFS-bisect heuristic (for abstract-infeasibility refinement)
# ---------------------------------------------------------------------------

def bfs_bisect(block: frozenset[int], territory: Territory) -> frozenset[int]:
    """Split *block* at the BFS midpoint, returning the first half as a deviation set.

    Performs a breadth-first traversal of the sectors in *block* starting from
    the lexicographically smallest sector, using concrete 4-adjacency restricted
    to the block.  Sectors are ranked by BFS discovery order and the first
    ``⌊|block|/2⌋`` sectors form the returned set.

    The BFS ordering tends to produce spatially contiguous halves, reducing
    spurious abstract adjacencies after refinement.
    """
    if len(block) <= 1:
        return block

    sectors = territory.ordered_sectors()
    index_by_coord = {c: i for i, c in enumerate(sectors)}

    start = min(block)
    order: list[int] = []
    visited: set[int] = set()
    queue: deque[int] = deque([start])
    visited.add(start)

    while queue:
        u = queue.popleft()
        order.append(u)
        coord_u = sectors[u]
        for nb in neighbors4(coord_u):
            j = index_by_coord.get(nb)
            if j is not None and j in block and j not in visited:
                visited.add(j)
                queue.append(j)

    # If the block is disconnected, BFS may not visit all sectors.
    # Append any unvisited sectors in sorted order.
    for i in sorted(block):
        if i not in visited:
            order.append(i)

    mid = len(order) // 2
    return frozenset(order[:mid])
