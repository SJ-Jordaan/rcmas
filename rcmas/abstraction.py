"""Definitions 8-12 (EUMAS): Territory partition and abstract RCMAS.

Provides the data structures and operations for CEGAR-based abstraction
refinement of Nash equilibrium synthesis:

* :class:`Partition` — a partition of concrete sectors into blocks (Def 8)
* :class:`AbstractRCMAS` — the abstract game with weighted blocks (Def 9)
* :func:`lift_strategy` — map abstract solution to concrete strategy (Def 10)
* :func:`compute_deviation_set` — sectors differing between two terminal states (Def 11)
* :func:`refine_partition` — split blocks by deviation set (Def 12)
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Coord, State, Territory, largest_region_size, neighbors4
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
) -> LiftedStrategy:
    """Lift an abstract SMT solution to a concrete strategy (Def 10).

    1. Derive concrete terminal ownership from abstract terminal ownership:
       ``owner[i] = abstract_owner[membership[i]]``
    2. Expand abstract round-by-round actions to concrete actions by
       distributing sectors within each block across rounds.
    3. Compute concrete rewards using the model's largest-region-size.
    """
    partition = abstract_rcmas.partition
    concrete_territory = abstract_rcmas.concrete_territory
    num_agents = abstract_rcmas.num_agents
    S_concrete = len(concrete_territory)

    assert abstract_solution.final_state is not None
    assert abstract_solution.actions_by_round is not None

    abstract_owner = abstract_solution.final_state.owner_by_index

    # 1. Concrete terminal ownership
    terminal_owner = tuple(
        abstract_owner[partition.membership[i]] for i in range(S_concrete)
    )

    # 2. Build concrete actions round by round
    # For each agent, collect which abstract blocks they claim each round,
    # then expand to concrete sectors.
    abstract_actions = abstract_solution.actions_by_round
    abstract_horizon = len(abstract_actions)

    # Collect per-agent assignment: list of (round, block_index)
    agent_blocks: list[list[int]] = [[] for _ in range(num_agents)]
    for t in range(abstract_horizon):
        for a in range(num_agents):
            action_coord = abstract_actions[t][a]
            if action_coord is not None:
                block_idx = abstract_rcmas.territory.index_of(action_coord)
                if block_idx is not None:
                    agent_blocks[a].append(block_idx)

    # Expand each agent's block assignments to concrete sectors
    # Within each block, assign sectors in sorted order
    concrete_sectors = concrete_territory.ordered_sectors()
    agent_concrete_actions: list[list[Coord]] = [[] for _ in range(num_agents)]
    for a in range(num_agents):
        for block_idx in agent_blocks[a]:
            block_sectors = sorted(partition.blocks[block_idx])
            for si in block_sectors:
                agent_concrete_actions[a].append(concrete_sectors[si])

    # Determine concrete horizon
    max_actions = max((len(acs) for acs in agent_concrete_actions), default=0)
    concrete_horizon = max_actions if max_actions > 0 else 1

    # Build actions_by_round
    actions_by_round: list[tuple[Coord | None, ...]] = []
    for t in range(concrete_horizon):
        step: list[Coord | None] = []
        for a in range(num_agents):
            if t < len(agent_concrete_actions[a]):
                step.append(agent_concrete_actions[a][t])
            else:
                step.append(None)
        actions_by_round.append(tuple(step))

    # 3. Compute concrete rewards
    terminal_state = State(
        territory=concrete_territory,
        num_agents=num_agents,
        owner_by_index=terminal_owner,
        round_index=concrete_horizon,
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
