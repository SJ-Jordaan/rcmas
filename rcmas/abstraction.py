"""Territory partitions, the abstract RCMAS and lifting (thesis Sections 6.1-6.3, 6.5).

Numbering follows :mod:`rcmas.model`: sectors ``0..|T|-1`` in the enumeration
of Remark 3.1.1 (bottom row first), agents ``0..n-1`` and ``UNOWNED`` (``-1``)
for an unoccupied sector or block.  Blocks are numbered ``0..k-1`` in
increasing order of their smallest sectors (Section 6.1), so tool block ``p``
is the thesis block ``B_{p+1}``, and the canonical completion of an abstract
play (Definition 4.4.2) uses this order.

Contents:

- :class:`Partition` (Definition 6.1.1), with ``membership[i]`` the tool
  index of ``mu_P(i)``; the discrete partition ``P_bot``, the coarsest
  partition ``{T}``, the orbit partition ``T/H_geo`` (Section 6.5) and the
  rectangular, strip and BFS partitions that :func:`make_partition` names;
- :class:`AbstractRCMAS` and :func:`build_abstract_rcmas` (Definition
  6.1.2): block weights ``omega(B_p) = |B_p|``, block adjacency, the horizon
  ``m_hat = floor(k/n)`` and the abstract reward, the largest total weight of
  a component under block adjacency (:meth:`AbstractRCMAS.rewards_of`;
  :func:`rcmas.model.scores` does not apply to the abstract game);
- :func:`weighted_graph_symmetry`, :func:`abstract_symmetry` and
  :func:`abstract_symmetry_constraints` (Section 6.5): orbits of the
  automorphisms of the weighted abstract graph, and the lex-leader and
  spatial canonicalisation constraints built from them for the initial
  query of an abstract synthesis call;
- :func:`lift_play` (Definition 6.2.1) and :func:`is_balanced` (Lemma
  6.2.1);
- :func:`deviation_set` (Definition 6.3.2), :func:`refine_partition`
  (Definition 6.3.3) and :func:`bfs_bisect` (the BFS-bisect heuristic of
  Section 6.3).

The synthetic territory of an abstract RCMAS has one coordinate per block
only to fix the enumeration of the blocks.  Its coordinates carry no
adjacency: every encoding of the abstract game takes the block adjacency and
the block weights as parameters (Section 7.1), and no symmetry is computed
from the synthetic coordinates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .model import UNOWNED, Coord, Territory, rewards_from_owner
from .strategy import Play

# ---------------------------------------------------------------------------
# Definition 6.1.1: territory partitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Partition:
    """A partition ``P = {B_1, ..., B_k}`` of the sectors (Definition 6.1.1).

    Attributes:
        blocks: the blocks as sets of sector indices, in increasing order of
            their smallest sectors.
        membership: ``membership[i]`` is the index of the block of sector
            ``i`` (the surjection ``mu_P``, 0-based).

    Use :meth:`from_blocks` to build a partition from any iterable of
    blocks; the constructor checks that the blocks are non-empty, disjoint,
    cover ``0..|T|-1`` and are numbered by their smallest sectors.
    """

    blocks: tuple[frozenset[int], ...]
    membership: tuple[int, ...]

    def __post_init__(self) -> None:
        seen: set[int] = set()
        last_min = -1
        for p, block in enumerate(self.blocks):
            if not block:
                raise ValueError("a block of a partition is non-empty")
            if seen & block:
                raise ValueError("the blocks of a partition are disjoint")
            if min(block) <= last_min:
                raise ValueError("blocks are numbered by their smallest sectors")
            last_min = min(block)
            seen |= block
            for i in block:
                if not 0 <= i < len(self.membership) or self.membership[i] != p:
                    raise ValueError("membership does not match the blocks")
        if len(seen) != len(self.membership):
            raise ValueError("the blocks of a partition cover the territory")

    @staticmethod
    def from_blocks(blocks: Iterable[Iterable[int]], num_sectors: int) -> Partition:
        """The partition with *blocks*, renumbered by their smallest sectors."""
        bs = sorted((frozenset(b) for b in blocks), key=lambda b: min(b) if b else -1)
        membership = [-1] * num_sectors
        for p, block in enumerate(bs):
            for i in block:
                if not 0 <= i < num_sectors:
                    raise ValueError(f"sector index {i} out of range")
                membership[i] = p
        return Partition(tuple(bs), tuple(membership))

    @property
    def num_sectors(self) -> int:
        """``|T|``."""
        return len(self.membership)

    def __len__(self) -> int:
        """``|P|``, the number ``k`` of blocks."""
        return len(self.blocks)

    @property
    def is_discrete(self) -> bool:
        """True for ``P_bot``, where every block is a single sector."""
        return len(self.blocks) == len(self.membership)

    def largest_block(self, candidates: Iterable[int] | None = None) -> int:
        """Index of a largest block among *candidates* (all blocks if ``None``).

        Ties are broken by block index, as in Example 6.3.2.
        """
        idx = list(range(len(self.blocks)) if candidates is None else candidates)
        if not idx:
            raise ValueError("no candidate block")
        return min(idx, key=lambda p: (-len(self.blocks[p]), p))


def discrete_partition(territory: Territory) -> Partition:
    """The discrete partition ``P_bot = {{i} | i in T}``."""
    S = len(territory)
    return Partition(tuple(frozenset({i}) for i in range(S)), tuple(range(S)))


def trivial_partition(territory: Territory) -> Partition:
    """The coarsest partition ``{T}`` (one block)."""
    S = len(territory)
    if S == 0:
        return Partition((), ())
    return Partition((frozenset(range(S)),), (0,) * S)


def orbit_partition(territory: Territory) -> Partition:
    """The orbit partition ``T/H_geo`` (Sections 6.1 and 6.5).

    Its blocks are the sector orbits under the geometric subgroup that
    symmetry detection computes (Section 7.1).  On a territory whose
    geometric subgroup is trivial it is the discrete partition.
    """
    from .symmetry import sector_orbits, territory_automorphisms

    orbits = sector_orbits(territory, territory_automorphisms(territory))
    return Partition.from_blocks(orbits, len(territory))


def grid_partition(territory: Territory, tile_w: int, tile_h: int) -> Partition:
    """Partition the territory into rectangular tiles of *tile_w* x *tile_h* cells.

    Sector ``(x, y)`` belongs to tile ``(x // tile_w, y // tile_h)``, with
    the coordinates of the grid file.  Empty tiles are omitted.
    """
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("tile dimensions must be >= 1")
    tiles: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(territory.ordered_sectors()):
        tiles.setdefault((c.x // tile_w, c.y // tile_h), []).append(i)
    return Partition.from_blocks(tiles.values(), len(territory))


def balanced_partition(territory: Territory, n: int) -> Partition:
    """Partition the territory into *n* connected blocks of nearly equal size.

    A full rectangular grid is cut into *n* vertical strips (or horizontal
    strips if it is narrower than *n*).  Other territories fall back to
    :func:`bfs_kway_partition` with ``k = n``.
    """
    sectors = territory.ordered_sectors()
    S = len(sectors)
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n >= S:
        return discrete_partition(territory)
    xs = sorted({c.x for c in sectors})
    ys = sorted({c.y for c in sectors})
    if S == len(xs) * len(ys):
        if len(xs) >= n:
            return _strip_partition(territory, n, "x", xs)
        if len(ys) >= n:
            return _strip_partition(territory, n, "y", ys)
    return bfs_kway_partition(territory, n)


def _strip_partition(
    territory: Territory, n: int, axis: str, coords: list[int]
) -> Partition:
    """Split the sectors into *n* contiguous strips along *axis*."""
    strip_of = {c: i * n // len(coords) for i, c in enumerate(coords)}
    strips: dict[int, list[int]] = {}
    for idx, sec in enumerate(territory.ordered_sectors()):
        strips.setdefault(strip_of[sec.x if axis == "x" else sec.y], []).append(idx)
    return Partition.from_blocks(strips.values(), len(territory))


def bfs_kway_partition(territory: Territory, k: int) -> Partition:
    """Partition the territory into *k* blocks by simultaneous BFS from *k* seeds.

    The seeds are spread evenly over the sector enumeration, all of them
    grow one hop per round, and each sector joins the first seed that
    reaches it (smaller seed index on ties).  Sectors that no seed reaches
    (on a disconnected territory) form one further block.
    """
    S = len(territory)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k >= S:
        return discrete_partition(territory)
    nbrs = territory.neighbor_map()
    seeds = [i * S // k for i in range(k)]
    assignment = [-1] * S
    frontiers: list[list[int]] = []
    for s_idx, s in enumerate(seeds):
        assignment[s] = s_idx
        frontiers.append([s])
    while any(frontiers):
        for s_idx in range(k):
            nxt: list[int] = []
            for u in frontiers[s_idx]:
                for j in sorted(nbrs[u]):
                    if assignment[j] == -1:
                        assignment[j] = s_idx
                        nxt.append(j)
            frontiers[s_idx] = nxt
    groups: dict[int, list[int]] = {}
    for i, s_idx in enumerate(assignment):
        groups.setdefault(s_idx, []).append(i)
    return Partition.from_blocks(groups.values(), S)


def make_partition(
    spec: str | Partition, territory: Territory, num_agents: int
) -> Partition:
    """The initial partition named by *spec*, or *spec* itself if it is a partition.

    Names: ``"orbit"`` (``T/H_geo``), ``"discrete"`` (``P_bot``),
    ``"trivial"`` (``{T}``), ``"balanced"``, ``"grid-WxH"`` and ``"bfs-K"``.
    """
    if isinstance(spec, Partition):
        if spec.num_sectors != len(territory):
            raise ValueError("the partition does not match the territory")
        return spec
    if spec == "orbit":
        return orbit_partition(territory)
    if spec == "discrete":
        return discrete_partition(territory)
    if spec in ("trivial", "coarsest"):
        return trivial_partition(territory)
    if spec == "balanced":
        return balanced_partition(territory, num_agents)
    if spec.startswith("grid-"):
        try:
            tw, th = (int(x) for x in spec[5:].split("x"))
        except ValueError:
            raise ValueError(
                f"invalid grid partition format: {spec!r} (expected grid-WxH)"
            ) from None
        return grid_partition(territory, tw, th)
    if spec.startswith("bfs-"):
        try:
            k = int(spec[4:])
        except ValueError:
            raise ValueError(
                f"invalid bfs partition format: {spec!r} (expected bfs-K)"
            ) from None
        return bfs_kway_partition(territory, k)
    raise ValueError(f"unknown partition type: {spec!r}")


# ---------------------------------------------------------------------------
# Definition 6.1.2: the abstract RCMAS
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AbstractRCMAS:
    """The abstract RCMAS ``M_hat_P = (Agt, T_hat, d)`` of a partition (Definition 6.1.2).

    Attributes:
        partition: the partition ``P``.
        territory: synthetic territory with one coordinate per block, in
            block order; it only fixes the enumeration of the blocks.
        weights: ``weights[p] = omega(B_p) = |B_p|``.
        neighbors: block adjacency ``B_p ~ B_q`` (index -> neighbour indices).
        horizon: ``m_hat = floor(k / n)``.
        num_agents: ``|Agt|``.
        concrete_territory: the territory ``T`` of ``M``.
    """

    partition: Partition
    territory: Territory
    weights: tuple[int, ...]
    neighbors: dict[int, frozenset[int]] = field(hash=False, compare=False)
    horizon: int
    num_agents: int
    concrete_territory: Territory

    @property
    def num_blocks(self) -> int:
        """``k = |P|``."""
        return len(self.weights)

    @property
    def concrete_horizon(self) -> int:
        """``m = floor(|T| / n)`` of the concrete game."""
        return len(self.concrete_territory) // self.num_agents

    @property
    def is_discrete(self) -> bool:
        """True if ``P`` is the discrete partition, so the abstract game is ``M``."""
        return self.partition.is_discrete

    @property
    def custom_neighbors(self) -> dict[int, list[int]]:
        """The block adjacency in the form that the encoding builders take."""
        return {p: sorted(self.neighbors.get(p, ())) for p in range(self.num_blocks)}

    def rewards_of(self, owner_by_block: Sequence[int]) -> tuple[int, ...]:
        """Abstract rewards of a final abstract state (Definition 6.1.2(4)).

        Agent ``a``'s reward is the largest total weight of a connected set
        of its blocks under the block adjacency.
        """
        return rewards_from_owner(
            tuple(owner_by_block), self.num_agents, self.neighbors, self.weights
        )


def block_adjacency(
    territory: Territory, partition: Partition
) -> dict[int, frozenset[int]]:
    """``B_p ~ B_q`` iff ``p != q`` and some ``i in B_p``, ``j in B_q`` have ``i ~ j``."""
    nbrs = territory.neighbor_map()
    out: dict[int, set[int]] = {p: set() for p in range(len(partition))}
    for i, js in nbrs.items():
        p = partition.membership[i]
        for j in js:
            q = partition.membership[j]
            if p != q:
                out[p].add(q)
    return {p: frozenset(qs) for p, qs in out.items()}


def build_abstract_rcmas(
    territory: Territory,
    num_agents: int,
    partition: Partition,
) -> AbstractRCMAS:
    """Construct ``M_hat_P`` from the concrete territory and a partition (Definition 6.1.2)."""
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if partition.num_sectors != len(territory):
        raise ValueError("the partition does not match the territory")
    k = len(partition)
    coords = tuple(Coord(p, 0) for p in range(k))
    synthetic = Territory(
        frozenset(coords), coords, {c: p for p, c in enumerate(coords)}
    )
    return AbstractRCMAS(
        partition=partition,
        territory=synthetic,
        weights=tuple(len(b) for b in partition.blocks),
        neighbors=block_adjacency(territory, partition),
        horizon=k // num_agents,
        num_agents=num_agents,
        concrete_territory=territory,
    )


# ---------------------------------------------------------------------------
# Section 6.5: symmetries of the weighted abstract graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AbstractSymmetry:
    """Orbits of a group of automorphisms of a weighted graph (Section 6.5).

    Attributes:
        generators: automorphisms (permutations ``p -> sigma(p)``) found by
            the search; the orbits are those of the group they generate.
        orbits: the orbits, sorted by their smallest element.
        representatives: the smallest element of each orbit.

    With an unbounded search the group is the full automorphism group of
    the graph.  If the search budget runs out for a pair of vertices, the
    group can be a proper subgroup, which Remark 5.1.1 allows.
    """

    generators: tuple[dict[int, int], ...]
    orbits: tuple[frozenset[int], ...]
    representatives: tuple[int, ...]

    @property
    def is_trivial(self) -> bool:
        """True if every orbit is a single vertex."""
        return all(len(o) == 1 for o in self.orbits)


class _Budget:
    __slots__ = ("left",)

    def __init__(self, nodes: int) -> None:
        self.left = nodes


def _refine_pair(
    adj: Sequence[Sequence[int]], c1: list[int], c2: list[int]
) -> tuple[list[int], list[int]] | None:
    """Colour refinement of two colourings in lockstep.

    Returns the stable refinements, relabelled consistently, or ``None`` if
    the colour classes stop matching (no colour-preserving isomorphism maps
    the first colouring to the second).
    """
    k = len(adj)
    while True:
        s1 = [(c1[x], tuple(sorted(c1[y] for y in adj[x]))) for x in range(k)]
        s2 = [(c2[x], tuple(sorted(c2[y] for y in adj[x]))) for x in range(k)]
        if sorted(s1) != sorted(s2):
            return None
        label = {s: r for r, s in enumerate(sorted(set(s1)))}
        n1 = [label[s] for s in s1]
        n2 = [label[s] for s in s2]
        if len(label) == len(set(c1)):
            return n1, n2
        c1, c2 = n1, n2


def _search(
    adj: Sequence[Sequence[int]], c1: list[int], c2: list[int], budget: _Budget
) -> dict[int, int] | None:
    """Individualisation-refinement search for an isomorphism from ``c1`` to ``c2``."""
    budget.left -= 1
    if budget.left < 0:
        return None
    refined = _refine_pair(adj, c1, c2)
    if refined is None:
        return None
    c1, c2 = refined
    k = len(adj)
    cells: dict[int, list[int]] = {}
    for x in range(k):
        cells.setdefault(c1[x], []).append(x)
    open_cells = [col for col, xs in cells.items() if len(xs) > 1]
    if not open_cells:
        image = {c2[y]: y for y in range(k)}
        perm = {x: image[c1[x]] for x in range(k)}
        for x in range(k):
            if {perm[y] for y in adj[x]} != set(adj[perm[x]]):
                return None
        return perm
    col = min(open_cells, key=lambda c: (len(cells[c]), c))
    x = cells[col][0]
    top = max(c1) + 1
    for y in sorted(y for y in range(k) if c2[y] == col):
        d1, d2 = list(c1), list(c2)
        d1[x] = top
        d2[y] = top
        found = _search(adj, d1, d2, budget)
        if found is not None:
            return found
        if budget.left < 0:
            return None
    return None


def weighted_graph_symmetry(
    neighbors: Mapping[int, Iterable[int]],
    weights: Sequence[int],
    *,
    budget: int = 20000,
) -> AbstractSymmetry:
    """Orbits of the automorphisms of a weighted graph on ``0..k-1``.

    An automorphism is a permutation of the vertices that preserves the
    adjacency *neighbors* and the weights (Section 6.5).  For every pair of
    vertices that colour refinement does not separate, an exhaustive
    individualisation-refinement search looks for an automorphism mapping
    one to the other, with at most *budget* search nodes per pair; the
    orbits are merged along every automorphism found.
    """
    k = len(weights)
    adj = [sorted(set(neighbors.get(p, ()))) for p in range(k)]
    for p in range(k):
        for q in adj[p]:
            if not 0 <= q < k or p not in adj[q] or q == p:
                raise ValueError("neighbors must be a symmetric irreflexive relation")
    if k == 0:
        return AbstractSymmetry((), (), ())
    wl = {w: r for r, w in enumerate(sorted(set(weights)))}
    base = [wl[w] for w in weights]
    refined = _refine_pair(adj, base, list(base))
    assert refined is not None
    colours = refined[0]

    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    gens: list[dict[int, int]] = []
    for v in range(k):
        if find(v) != v:
            continue
        for u in range(v):
            if colours[u] != colours[v] or find(u) != u:
                continue
            c1, c2 = list(colours), list(colours)
            top = max(colours) + 1
            c1[u] = top
            c2[v] = top
            perm = _search(adj, c1, c2, _Budget(budget))
            if perm is None:
                continue
            gens.append(perm)
            for x, y in perm.items():
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
            break
    groups: dict[int, set[int]] = {}
    for x in range(k):
        groups.setdefault(find(x), set()).add(x)
    orbits = tuple(sorted((frozenset(g) for g in groups.values()), key=min))
    return AbstractSymmetry(
        generators=tuple(gens),
        orbits=orbits,
        representatives=tuple(sorted(min(o) for o in orbits)),
    )


def abstract_symmetry(abstract: AbstractRCMAS) -> AbstractSymmetry:
    """Orbits of the automorphisms of the weighted abstract graph of *abstract*.

    These are the permutations of the blocks that preserve the block
    adjacency and the weights (Section 6.5).  They are computed from the
    abstract graph itself, never from the coordinates of the synthetic
    territory.
    """
    return weighted_graph_symmetry(abstract.neighbors, abstract.weights)


def abstract_symmetry_constraints(
    abstract: AbstractRCMAS,
    demands: tuple[int, ...] | None = None,
    symmetry: AbstractSymmetry | None = None,
) -> Callable[[Any, Any], None]:
    """The symmetry-breaking constraints of the abstract game (Section 6.5).

    Returns a function ``(solver, variables) -> None`` that adds the
    lex-leader constraint within the demand classes of *demands*
    (Definition 5.1.6) and the spatial canonicalisation constraint with the
    orbit representatives of the weighted abstract graph (Definition 5.1.7).
    It is meant for the query that produces the initial play of the inner
    synthesiser only (``initial_constraints`` of IBIS, S-IBIS and Q-IBIS),
    where Theorem 5.1.3 applies, since these permutations preserve the
    abstract dynamics and the weighted rewards.
    """
    from .smt_constraints import symmetry_breaking_constraint
    from .symmetry import SymmetryInfo

    sym = abstract_symmetry(abstract) if symmetry is None else symmetry
    identity = {p: p for p in range(abstract.num_blocks)}
    # The constraint reads only the representatives; the generators are
    # recorded for inspection.
    info = SymmetryInfo(
        automorphisms=(identity, *sym.generators),
        orbits=sym.orbits,
        representatives=sym.representatives,
    )

    def add(opt: Any, v: Any) -> None:
        symmetry_breaking_constraint(opt, v, info, demands=demands)

    return add


# ---------------------------------------------------------------------------
# Definition 6.2.1: lifting
# ---------------------------------------------------------------------------


def lift_play(abstract: AbstractRCMAS, abstract_play: Play) -> Play:
    """The lifted play ``lift_P(alpha_hat)`` of Definition 6.2.1.

    *abstract_play* is the outcome of the abstract strategy, a play of the
    abstract game with ``m_hat`` rounds over the block indices.  The result
    is a play of ``M`` with ``m = floor(|T|/n)`` rounds:

    1. each agent inherits the sectors of the blocks it claims;
    2. an agent whose blocks hold more than ``m`` sectors keeps the first
       ``m`` of them, in the temporal order of its block claims and in
       increasing index order within a block;
    3. the remaining unclaimed sectors, in increasing index order, go one
       by one to the agent that holds the fewest sectors (smallest index on
       ties) until every agent holds ``m``;
    4. in round ``t`` every agent claims the ``t``-th of its sectors, the
       sectors of its blocks first, in the order of step 2, and the
       distributed sectors last, in increasing index order.
    """
    n = abstract.num_agents
    k = abstract.num_blocks
    S = len(abstract.concrete_territory)
    m = abstract.concrete_horizon
    if abstract_play.num_agents != n:
        raise ValueError("the abstract play has the wrong number of agents")
    if len(abstract_play.states[0]) != k or abstract_play.horizon != abstract.horizon:
        raise ValueError("the abstract play is not a play of the abstract game")
    blocks = abstract.partition.blocks

    # Steps 1 and 2.
    claims: list[list[int]] = []
    for a in range(n):
        seq = [i for p in abstract_play.claims_of(a) for i in sorted(blocks[p])]
        claims.append(seq[:m])
    held = {i for seq in claims for i in seq}

    # Step 3.
    for i in range(S):
        if all(len(seq) == m for seq in claims):
            break
        if i in held:
            continue
        a = min(range(n), key=lambda b: (len(claims[b]), b))
        claims[a].append(i)
        held.add(i)

    # Step 4.
    return Play.from_actions(S, n, [[claims[a][t] for a in range(n)] for t in range(m)])


def is_balanced(abstract: AbstractRCMAS, abstract_play: Play) -> bool:
    """True if every agent's claimed blocks hold exactly ``m`` sectors (Lemma 6.2.1)."""
    m = abstract.concrete_horizon
    return all(
        sum(abstract.weights[p] for p in abstract_play.claims_of(a)) == m
        for a in range(abstract.num_agents)
    )


@dataclass(frozen=True, slots=True)
class LiftedStrategy:
    """A lifted play with its final state and concrete rewards (legacy container).

    Attributes:
        terminal_owner: the final control vector ``s_m``.
        actions_by_round: the profiles of the play, as coordinates.
        payoff_by_agent: ``r_a`` on ``s_m`` (Definition 3.9.2).
        play: the lifted play.
    """

    terminal_owner: tuple[int, ...]
    actions_by_round: tuple[tuple[Coord | None, ...], ...]
    payoff_by_agent: tuple[int, ...]
    play: Play | None = None


def lift_strategy(
    abstract_rcmas: AbstractRCMAS,
    abstract_solution: Any,
    concrete_horizon: int | None = None,
) -> LiftedStrategy:
    """Lift the play of an abstract SMT solution (Definition 6.2.1).

    Wrapper around :func:`lift_play` for callers that hold an
    :class:`~rcmas.smt_solve.SmtSolution` of the abstract game.
    *concrete_horizon* must be ``None`` or ``m``.
    """
    if (
        concrete_horizon is not None
        and concrete_horizon != abstract_rcmas.concrete_horizon
    ):
        raise ValueError("the lifted play has m = floor(|T|/n) rounds")
    abstract_play = abstract_solution.play
    if abstract_play is None:
        abstract_play = Play.from_coord_actions(
            abstract_rcmas.territory,
            abstract_rcmas.num_agents,
            abstract_solution.actions_by_round,
        )
    play = lift_play(abstract_rcmas, abstract_play)
    concrete = abstract_rcmas.concrete_territory
    return LiftedStrategy(
        terminal_owner=play.final_state,
        actions_by_round=play.to_coords(concrete),
        payoff_by_agent=play.rewards(concrete.neighbor_map()),
        play=play,
    )


# ---------------------------------------------------------------------------
# Definitions 6.3.2 and 6.3.3: deviation sets and refinement
# ---------------------------------------------------------------------------


def deviation_set(
    final_state: Sequence[int], deviated_final_state: Sequence[int]
) -> frozenset[int]:
    """``D_a = {i | s_m(i) != s'_m(i)}`` (Definition 6.3.2)."""
    if len(final_state) != len(deviated_final_state):
        raise ValueError("the states belong to different territories")
    return frozenset(
        i for i, (o, o2) in enumerate(zip(final_state, deviated_final_state)) if o != o2
    )


compute_deviation_set = deviation_set
"""Alias of :func:`deviation_set`."""


def sectors_of(final_state: Sequence[int], agent: int) -> frozenset[int]:
    """``{i | s_m(i) = agent}``, the refinement set of an unsatisfying play (Section 6.3)."""
    if agent == UNOWNED:
        raise ValueError("agent must be an agent index")
    return frozenset(i for i, o in enumerate(final_state) if o == agent)


def refine_partition(partition: Partition, deviation: Iterable[int]) -> Partition:
    """``Refine(P, D)`` of Definition 6.3.3.

    Every block ``B`` with ``B & D`` and ``B - D`` both non-empty is split
    into these two blocks; the other blocks are kept.  The result is
    numbered by the smallest sectors of its blocks.
    """
    D = frozenset(deviation)
    out: list[frozenset[int]] = []
    for block in partition.blocks:
        inside, outside = block & D, block - D
        if inside and outside:
            out.extend((inside, outside))
        else:
            out.append(block)
    return Partition.from_blocks(out, partition.num_sectors)


def bfs_bisect(block: Iterable[int], territory: Territory) -> frozenset[int]:
    """The first half of *block* in BFS order (the BFS-bisect heuristic of Section 6.3).

    A breadth-first traversal of the subgraph that the block induces starts
    at its smallest sector and visits neighbours in increasing index order;
    sectors it does not reach follow in increasing index order.  The first
    ``floor(|B|/2)`` sectors of this ranking are returned, so
    ``Refine(P, bfs_bisect(B))`` splits a block of at least two sectors
    into two non-empty blocks and leaves every other block unchanged.
    """
    B = frozenset(block)
    if not B:
        return frozenset()
    nbrs = territory.neighbor_map()
    start = min(B)
    order = [start]
    visited = {start}
    queue: deque[int] = deque([start])
    while queue:
        u = queue.popleft()
        for j in sorted(nbrs[u]):
            if j in B and j not in visited:
                visited.add(j)
                order.append(j)
                queue.append(j)
    order.extend(i for i in sorted(B) if i not in visited)
    return frozenset(order[: len(order) // 2])
