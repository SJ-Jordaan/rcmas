"""Tests for rcmas.abstraction (thesis Sections 6.1-6.3 and 6.5).

Grid files list rows from the top, and the tool numbers sectors from the
bottom row (Remark 3.1.1).  The examples of Chapter 6 draw sector 1 in the
top-left corner; all of them are symmetric under the top-bottom reflection,
so thesis sector ``L`` is tool sector ``L - 1`` in the reflected picture.
"""

from __future__ import annotations

import itertools
import random

import pytest

from rcmas.abstraction import (
    AbstractRCMAS,
    Partition,
    abstract_symmetry,
    abstract_symmetry_constraints,
    balanced_partition,
    bfs_bisect,
    bfs_kway_partition,
    block_adjacency,
    build_abstract_rcmas,
    deviation_set,
    discrete_partition,
    grid_partition,
    is_balanced,
    lift_play,
    lift_strategy,
    make_partition,
    orbit_partition,
    refine_partition,
    sectors_of,
    trivial_partition,
    weighted_graph_symmetry,
)
from rcmas.model import UNOWNED, Territory, rewards_from_owner
from rcmas.strategy import Play

from .brute_force import is_equilibrium, random_play

GRID_4x2 = ["....", "...."]
GRID_6x2 = ["......", "......"]
GRID_3x3 = ["...", "...", "..."]


def _t(rows: list[str]) -> Territory:
    return Territory.from_ascii(rows)


def _random_partition(territory: Territory, k: int, rng: random.Random) -> Partition:
    labels = [rng.randrange(k) for _ in range(len(territory))]
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    return Partition.from_blocks(groups.values(), len(territory))


def _components(owner: tuple[int, ...], agent: int, nbrs: dict) -> list[set[int]]:
    seen: set[int] = set()
    out = []
    for s, o in enumerate(owner):
        if o != agent or s in seen:
            continue
        comp = {s}
        stack = [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            for w in nbrs[u]:
                if w not in seen and owner[w] == agent:
                    seen.add(w)
                    comp.add(w)
                    stack.append(w)
        out.append(comp)
    return out


# ---------------------------------------------------------------------------
# Definition 6.1.1: partitions
# ---------------------------------------------------------------------------


class TestPartition:
    def test_from_blocks_numbers_by_smallest_sector(self) -> None:
        P = Partition.from_blocks([{3, 1}, {0, 2}], 4)
        assert P.blocks == (frozenset({0, 2}), frozenset({1, 3}))
        assert P.membership == (0, 1, 0, 1)
        assert len(P) == 2 and P.num_sectors == 4

    @pytest.mark.parametrize(
        "blocks, membership",
        [
            ((frozenset({0}), frozenset()), (0, 0)),  # empty block
            ((frozenset({0, 1}), frozenset({1})), (0, 1)),  # overlap
            ((frozenset({0}),), (0, 0)),  # not a cover
            ((frozenset({1}), frozenset({0})), (1, 0)),  # wrong numbering
            ((frozenset({0}), frozenset({1})), (0, 0)),  # wrong membership
        ],
    )
    def test_invalid_partitions_rejected(self, blocks, membership) -> None:
        with pytest.raises(ValueError):
            Partition(blocks, membership)

    def test_discrete_and_trivial(self, territory_3x3: Territory) -> None:
        d = discrete_partition(territory_3x3)
        assert d.is_discrete and len(d) == 9
        assert all(d.blocks[i] == {i} for i in range(9))
        t = trivial_partition(territory_3x3)
        assert len(t) == 1 and t.blocks[0] == frozenset(range(9))

    def test_orbit_partition_3x3(self, territory_3x3: Territory) -> None:
        P = orbit_partition(territory_3x3)
        # corners, edges, centre, numbered by their smallest sectors
        assert P.blocks == (
            frozenset({0, 2, 6, 8}),
            frozenset({1, 3, 5, 7}),
            frozenset({4}),
        )

    def test_orbit_partition_of_asymmetric_territory_is_discrete(self) -> None:
        t = _t(["..", ".", "..."])
        assert orbit_partition(t).is_discrete

    def test_largest_block_ties_by_index(self) -> None:
        P = Partition.from_blocks([{0, 1}, {2, 3}, {4}], 5)
        assert P.largest_block() == 0
        assert P.largest_block([1, 2]) == 1
        assert P.largest_block([2]) == 2
        with pytest.raises(ValueError):
            P.largest_block([])

    @pytest.mark.parametrize(
        "spec", ["orbit", "discrete", "trivial", "balanced", "grid-2x2", "bfs-3"]
    )
    def test_named_partitions_are_partitions(self, spec: str) -> None:
        t = _t(["....", "....", "..."])
        P = make_partition(spec, t, 2)
        assert P.num_sectors == len(t)
        assert sorted(i for b in P.blocks for i in b) == list(range(len(t)))

    def test_make_partition_errors(self, territory_2x2: Territory) -> None:
        for bad in ("bogus", "grid-2", "bfs-x"):
            with pytest.raises(ValueError):
                make_partition(bad, territory_2x2, 2)
        with pytest.raises(ValueError):
            make_partition(discrete_partition(_t(["..."])), territory_2x2, 2)

    def test_grid_partition_2x2_tiles_on_4x4(self) -> None:
        P = grid_partition(_t(["...."] * 4), 2, 2)
        assert len(P) == 4 and all(len(b) == 4 for b in P.blocks)

    def test_balanced_partition_strips(self) -> None:
        P = balanced_partition(_t(GRID_4x2), 2)
        assert P.blocks == (frozenset({0, 1, 4, 5}), frozenset({2, 3, 6, 7}))

    def test_bfs_kway_disconnected_keeps_unreached_block(self) -> None:
        P = bfs_kway_partition(_t(["..#.."]), 1)
        assert P.blocks == (frozenset({0, 1}), frozenset({2, 3}))
        P2 = bfs_kway_partition(_t(["..#....#.."]), 2)
        assert sorted(i for b in P2.blocks for i in b) == list(range(8))


# ---------------------------------------------------------------------------
# Definition 6.1.2: the abstract RCMAS
# ---------------------------------------------------------------------------


class TestAbstractRCMAS:
    def test_example_6_1_2_column_partition(self) -> None:
        """Example 6.1.2: the column partition of the 4x2 grid is a weighted path."""
        t = _t(GRID_4x2)
        P = Partition.from_blocks([{0, 4}, {1, 5}, {2, 6}, {3, 7}], 8)
        A = build_abstract_rcmas(t, 2, P)
        assert A.weights == (2, 2, 2, 2)
        assert A.neighbors == {
            0: frozenset({1}),
            1: frozenset({0, 2}),
            2: frozenset({1, 3}),
            3: frozenset({2}),
        }
        assert A.horizon == 2 and A.concrete_horizon == 4
        assert A.rewards_of((0, 0, 1, 1)) == (4, 4)

    def test_abstract_reward_uses_block_adjacency_and_weights(self) -> None:
        """The abstract reward is not model.scores on the synthetic row (Section 6.1)."""
        t = _t(GRID_3x3)
        # corners, edges, centre: corners ~ edges ~ centre, corners !~ centre
        P = orbit_partition(t)
        A = build_abstract_rcmas(t, 2, P)
        assert A.neighbors == {
            0: frozenset({1}),
            1: frozenset({0, 2}),
            2: frozenset({1}),
        }
        # Blocks 0 and 2 are next to each other in the synthetic row, but not
        # adjacent in the abstract game; weights count.
        assert A.rewards_of((0, 1, 0)) == (4, 4)
        assert A.rewards_of((0, 0, UNOWNED)) == (8, 0)

    def test_example_6_1_3_over_approximation(self) -> None:
        """Example 6.1.3: rows are a path; corners/edges over-approximate."""
        t = _t(GRID_3x3)
        rows = Partition.from_blocks([{0, 1, 2}, {3, 4, 5}, {6, 7, 8}], 9)
        A = build_abstract_rcmas(t, 3, rows)
        assert 2 not in A.neighbors[0] and A.neighbors[1] == {0, 2}
        B = build_abstract_rcmas(t, 2, orbit_partition(t))
        # Agent holds corners 1, 9 and edges 4, 6 (thesis labels).
        owner = [UNOWNED] * 9
        for i in (0, 8, 3, 5):
            owner[i] = 0
        assert rewards_from_owner(tuple(owner), 2, t.neighbor_map())[0] == 2
        assert B.rewards_of((0, 0, UNOWNED))[0] == 8

    def test_block_adjacency_matches_definition(self) -> None:
        rng = random.Random(1)
        t = _t(["....", "....", "..."])
        nbrs = t.neighbor_map()
        for _ in range(20):
            P = _random_partition(t, rng.randint(1, 6), rng)
            adj = block_adjacency(t, P)
            for p, q in itertools.permutations(range(len(P)), 2):
                expected = any(j in nbrs[i] for i in P.blocks[p] for j in P.blocks[q])
                assert (q in adj[p]) == expected

    def test_horizon_floor_k_over_n(self, territory_3x3: Territory) -> None:
        P = orbit_partition(territory_3x3)
        assert build_abstract_rcmas(territory_3x3, 2, P).horizon == 1
        assert build_abstract_rcmas(territory_3x3, 4, P).horizon == 0

    def test_synthetic_territory_numbering(self, territory_3x3: Territory) -> None:
        A = build_abstract_rcmas(territory_3x3, 2, orbit_partition(territory_3x3))
        assert len(A.territory) == 3
        assert [A.territory.index_of(c) for c in A.territory.ordered_sectors()] == [
            0,
            1,
            2,
        ]
        assert A.custom_neighbors == {0: [1], 1: [0, 2], 2: [1]}
        assert A.concrete_territory is territory_3x3

    def test_mismatched_partition_rejected(self, territory_2x2: Territory) -> None:
        with pytest.raises(ValueError):
            build_abstract_rcmas(territory_2x2, 2, discrete_partition(_t(["."])))
        with pytest.raises(ValueError):
            build_abstract_rcmas(territory_2x2, 0, discrete_partition(territory_2x2))


# ---------------------------------------------------------------------------
# Section 6.1: the inner synthesisers apply to the abstract game
# ---------------------------------------------------------------------------


class TestAbstractSynthesis:
    @pytest.mark.parametrize(
        "rows, n, blocks, demands",
        [
            (GRID_4x2, 2, [{0, 4}, {1, 5}, {2, 6}, {3, 7}], None),
            (GRID_3x3, 2, [{0, 1, 3}, {2, 5}, {4}, {6, 7}, {8}], None),
            (GRID_3x3, 2, [{0, 1, 3}, {2, 5}, {4}, {6, 7}, {8}], (2, 2)),
            (GRID_3x3, 3, [{0}, {1, 2}, {3, 6}, {4}, {5, 8}, {7}], None),
        ],
    )
    def test_sibis_equilibria_of_abstract_games(self, rows, n, blocks, demands) -> None:
        """S-IBIS on M_hat returns equilibria of the weighted abstract game."""
        pytest.importorskip("z3")
        from rcmas.ibis import solve_sibis

        t = _t(rows)
        A = build_abstract_rcmas(t, n, Partition.from_blocks(blocks, len(t)))
        res = solve_sibis(
            territory=A.territory,
            num_agents=n,
            weights=A.weights,
            custom_neighbors=A.custom_neighbors,
            demands=demands,
        )
        assert res.outcome == "ne"
        assert res.play is not None
        assert res.payoff_by_agent == A.rewards_of(res.play.final_state)
        assert is_equilibrium(res.play, A.neighbors, demands, A.weights)


# ---------------------------------------------------------------------------
# Section 6.5: symmetries of the weighted abstract graph
# ---------------------------------------------------------------------------


def _brute_orbits(neighbors: dict, weights: tuple[int, ...]) -> set[frozenset[int]]:
    k = len(weights)
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for perm in itertools.permutations(range(k)):
        if any(weights[perm[p]] != weights[p] for p in range(k)):
            continue
        if any(
            {perm[q] for q in neighbors.get(p, ())} != set(neighbors.get(perm[p], ()))
            for p in range(k)
        ):
            continue
        for p in range(k):
            a, b = find(p), find(perm[p])
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[int, set[int]] = {}
    for p in range(k):
        groups.setdefault(find(p), set()).add(p)
    return {frozenset(g) for g in groups.values()}


class TestAbstractSymmetry:
    def test_orbits_match_brute_force(self) -> None:
        rng = random.Random(7)
        for rows in (GRID_3x3, GRID_4x2, ["...", "..."], ["..#..", "..#.."]):
            t = _t(rows)
            for _ in range(15):
                P = _random_partition(t, rng.randint(1, 7), rng)
                A = build_abstract_rcmas(t, 2, P)
                sym = abstract_symmetry(A)
                assert set(sym.orbits) == _brute_orbits(A.neighbors, A.weights)
                assert sym.representatives == tuple(sorted(min(o) for o in sym.orbits))
                for g in sym.generators:
                    assert sorted(g.values()) == list(range(A.num_blocks))
                    for p in range(A.num_blocks):
                        assert A.weights[g[p]] == A.weights[p]
                        assert {g[q] for q in A.neighbors[p]} == A.neighbors[g[p]]

    def test_graph_not_synthetic_row(self) -> None:
        """A star 0-1, 0-2 swaps 1 and 2; the synthetic row 0-1-2 would swap 0 and 2."""
        sym = weighted_graph_symmetry({0: [1, 2], 1: [0], 2: [0]}, (2, 2, 2))
        assert set(sym.orbits) == {frozenset({0}), frozenset({1, 2})}
        # Weights break the symmetry.
        sym = weighted_graph_symmetry({0: [1, 2], 1: [0], 2: [0]}, (2, 2, 3))
        assert sym.is_trivial

    def test_4x4_orbit_partition_swaps_corners_and_centre(self) -> None:
        t = _t(["...."] * 4)
        A = build_abstract_rcmas(t, 2, orbit_partition(t))
        assert A.weights == (4, 8, 4)
        sym = abstract_symmetry(A)
        assert set(sym.orbits) == {frozenset({0, 2}), frozenset({1})}

    def test_invalid_graph_rejected(self) -> None:
        with pytest.raises(ValueError):
            weighted_graph_symmetry({0: [1], 1: []}, (1, 1))

    @pytest.mark.parametrize(
        "rows, n, blocks, demands",
        [
            (["...."] * 4, 2, None, None),
            (GRID_4x2, 2, [{0, 4}, {1, 5}, {2, 6}, {3, 7}], None),
            (GRID_4x2, 2, [{0, 4}, {1, 5}, {2, 6}, {3, 7}], (4, 4)),
            (GRID_3x3, 3, [{0, 2}, {1}, {3, 5}, {4}, {6, 8}, {7}], (2, 2, 1)),
            (["..#..", "..#.."], 2, [{0}, {1}, {2}, {3}, {4, 5}, {6, 7}], (2, 1)),
        ],
    )
    def test_constraints_preserve_feasibility_and_optimum(
        self, rows, n, blocks, demands
    ) -> None:
        """Corollary 5.1.4 for the abstract game: same satisfiability, same optimum."""
        pytest.importorskip("z3")
        from rcmas.smt_solve import (
            build_base_encoding,
            find_feasible_from_base,
            solve_from_base,
        )

        t = _t(rows)
        P = (
            orbit_partition(t)
            if blocks is None
            else Partition.from_blocks(blocks, len(t))
        )
        A = build_abstract_rcmas(t, n, P)
        base = build_base_encoding(
            territory=A.territory,
            num_agents=n,
            horizon=A.horizon,
            demands=demands,
            weights=A.weights,
            custom_neighbors=A.custom_neighbors,
        )
        sym = abstract_symmetry_constraints(A, demands)
        plain = find_feasible_from_base(base)
        reduced = find_feasible_from_base(base, extra_constraints=sym)
        assert plain.is_sat == reduced.is_sat
        if plain.is_sat:
            best = solve_from_base(base, objective="sum")
            best_sym = solve_from_base(base, objective="sum", extra_constraints=sym)
            assert sum(best.payoff_by_agent) == sum(best_sym.payoff_by_agent)


# ---------------------------------------------------------------------------
# Definition 6.2.1: lifting
# ---------------------------------------------------------------------------


class TestLifting:
    def test_example_6_2_1(self) -> None:
        """Example 6.2.1: column partition, agent 1 claims B1 then B2, agent 2 B4 then B3."""
        t = _t(GRID_4x2)
        A = build_abstract_rcmas(
            t, 2, Partition.from_blocks([{0, 4}, {1, 5}, {2, 6}, {3, 7}], 8)
        )
        abstract = Play.from_actions(4, 2, [[0, 3], [1, 2]])
        assert is_balanced(A, abstract)
        pi = lift_play(A, abstract)
        assert pi.actions == ((0, 3), (4, 7), (1, 2), (5, 6))
        assert pi.final_state == (0, 0, 1, 1, 0, 0, 1, 1)
        assert pi.rewards(t.neighbor_map()) == (4, 4)

    def test_trimming_and_distribution(self) -> None:
        """Steps 2-4: trim the first m sectors, then hand out the rest to the fewest-holder."""
        t = _t(["...", "..."])  # sectors 0,1,2 bottom row, 3,4,5 top row
        A = build_abstract_rcmas(
            t, 2, Partition.from_blocks([{0, 1, 2, 3}, {4}, {5}], 6)
        )
        abstract = Play.from_actions(3, 2, [[0, 1]])
        assert not is_balanced(A, abstract)
        pi = lift_play(A, abstract)
        # Agent 0 keeps 0, 1, 2; sector 3 reverts.  Agent 1 holds 4 and gets
        # 3 and then 5, claimed after its block sectors in index order.
        assert pi.claims_of(0) == (0, 1, 2)
        assert pi.claims_of(1) == (4, 3, 5)

    def test_zero_abstract_horizon_distributes_greedily(self) -> None:
        t = _t(GRID_6x2)
        A = build_abstract_rcmas(t, 3, trivial_partition(t))
        assert A.horizon == 0
        pi = lift_play(A, Play.from_actions(1, 3, []))
        assert pi.claims_of(0) == (0, 3, 6, 9)
        assert pi.claims_of(1) == (1, 4, 7, 10)
        assert pi.claims_of(2) == (2, 5, 8, 11)

    def test_lifted_plays_are_plays_with_m_rounds(self) -> None:
        rng = random.Random(3)
        for rows, n in ((GRID_3x3, 2), (GRID_3x3, 3), (GRID_4x2, 3), (GRID_6x2, 5)):
            t = _t(rows)
            m = len(t) // n
            for _ in range(30):
                A = build_abstract_rcmas(
                    t, n, _random_partition(t, rng.randint(1, 9), rng)
                )
                abstract = random_play(A.num_blocks, n, A.horizon, rng)
                pi = lift_play(A, abstract)
                assert pi.horizon == m
                Play.from_actions(len(t), n, pi.actions)  # collision-free play
                for a in range(n):
                    assert sum(1 for o in pi.final_state if o == a) == m

    def test_lemma_6_2_1_closed_form(self) -> None:
        rng = random.Random(5)
        t = _t(["....", "....", "...."])
        n = 2  # m = 6
        checked = 0
        for _ in range(200):
            blocks = [{0, 4}, {1, 5}, {2, 6}, {3, 7}, {8, 9}, {10, 11}]
            A = build_abstract_rcmas(t, n, Partition.from_blocks(blocks, 12))
            abstract = random_play(A.num_blocks, n, A.horizon, rng)
            assert is_balanced(A, abstract)
            pi = lift_play(A, abstract)
            for a in range(n):
                c_a = [
                    i
                    for p in abstract.claims_of(a)
                    for i in sorted(A.partition.blocks[p])
                ]
                assert list(pi.claims_of(a)) == c_a
            for i in range(12):
                assert (
                    pi.final_state[i] == abstract.final_state[A.partition.membership[i]]
                )
            checked += 1
        assert checked == 200

    def test_lemma_6_2_2_over_approximation(self) -> None:
        """Inherited components are no larger than the abstract reward."""
        rng = random.Random(11)
        for rows, n in ((GRID_3x3, 2), (["....", "....", "...."], 3), (GRID_6x2, 2)):
            t = _t(rows)
            nbrs = t.neighbor_map()
            for _ in range(40):
                A = build_abstract_rcmas(
                    t, n, _random_partition(t, rng.randint(2, 8), rng)
                )
                abstract = random_play(A.num_blocks, n, A.horizon, rng)
                pi = lift_play(A, abstract)
                r_hat = A.rewards_of(abstract.final_state)
                for a in range(n):
                    inherited = {
                        i for p in abstract.claims_of(a) for i in A.partition.blocks[p]
                    }
                    for comp in _components(pi.final_state, a, nbrs):
                        if comp <= inherited:
                            assert len(comp) <= r_hat[a]

    def test_discrete_lift_is_identity(self) -> None:
        rng = random.Random(2)
        t = _t(GRID_3x3)
        A = build_abstract_rcmas(t, 2, discrete_partition(t))
        for _ in range(10):
            play = random_play(9, 2, 4, rng)
            assert lift_play(A, play) == play

    def test_wrong_abstract_play_rejected(self) -> None:
        t = _t(GRID_4x2)
        A = build_abstract_rcmas(t, 2, trivial_partition(t))
        with pytest.raises(ValueError):
            lift_play(A, Play.from_actions(4, 2, [[0, 1]]))

    def test_lift_strategy_wrapper(self) -> None:
        pytest.importorskip("z3")
        from rcmas.smt_solve import build_base_encoding, find_feasible_from_base

        t = _t(GRID_4x2)
        A = build_abstract_rcmas(
            t, 2, Partition.from_blocks([{0, 4}, {1, 5}, {2, 6}, {3, 7}], 8)
        )
        base = build_base_encoding(
            territory=A.territory,
            num_agents=2,
            horizon=A.horizon,
            weights=A.weights,
            custom_neighbors=A.custom_neighbors,
        )
        sol = find_feasible_from_base(base)
        lifted = lift_strategy(A, sol, concrete_horizon=4)
        assert lifted.play == lift_play(A, sol.play)
        assert lifted.terminal_owner == lifted.play.final_state
        assert lifted.payoff_by_agent == lifted.play.rewards(t.neighbor_map())
        with pytest.raises(ValueError):
            lift_strategy(A, sol, concrete_horizon=3)


# ---------------------------------------------------------------------------
# Definitions 6.3.2, 6.3.3 and BFS-bisect
# ---------------------------------------------------------------------------


class TestRefinement:
    def test_deviation_set(self) -> None:
        assert deviation_set((0, 1, -1), (0, 1, -1)) == frozenset()
        assert deviation_set((0, 1, -1, 1), (1, 1, 0, 1)) == {0, 2}
        with pytest.raises(ValueError):
            deviation_set((0,), (0, 1))

    def test_sectors_of(self) -> None:
        assert sectors_of((0, 1, 0, -1), 0) == {0, 2}
        with pytest.raises(ValueError):
            sectors_of((0,), UNOWNED)

    def test_example_6_3_1(self) -> None:
        """Example 6.3.1: the half partition refined by the middle columns."""
        halves = Partition.from_blocks([{0, 1, 4, 5}, {2, 3, 6, 7}], 8)
        P = refine_partition(halves, {1, 2, 5, 6})
        assert P.blocks == (
            frozenset({0, 4}),
            frozenset({1, 5}),
            frozenset({2, 6}),
            frozenset({3, 7}),
        )

    def test_refine_splits_exactly_the_straddling_blocks(self) -> None:
        rng = random.Random(4)
        t = _t(GRID_3x3)
        for _ in range(50):
            P = _random_partition(t, rng.randint(1, 9), rng)
            D = frozenset(i for i in range(9) if rng.random() < 0.4)
            Q = refine_partition(P, D)
            straddling = sum(1 for B in P.blocks if B & D and B - D)
            assert len(Q) == len(P) + straddling
            for B in Q.blocks:
                assert B <= D or not (B & D)
                assert any(B <= C for C in P.blocks)

    def test_union_of_blocks_changes_nothing(self) -> None:
        P = Partition.from_blocks([{0, 1}, {2, 3}, {4}], 5)
        assert refine_partition(P, {0, 1, 4}) == P
        assert refine_partition(P, set()) == P

    def test_example_6_3_2_bfs_bisect(self) -> None:
        """Example 6.3.2: BFS from sector 1 ranks 1, 2, 7, 3, 8, 9; the split is {1, 2, 7}."""
        t = _t(GRID_6x2)
        assert bfs_bisect({0, 1, 2, 6, 7, 8}, t) == {0, 1, 6}

    def test_bfs_bisect_disconnected_block(self) -> None:
        t = _t(GRID_3x3)
        # BFS from 0 reaches 1; 8 is unreached and ranked last.
        assert bfs_bisect({0, 1, 8}, t) == {0}
        assert bfs_bisect({0, 8, 4, 2}, t) == {0, 2}

    def test_bfs_bisect_adds_one_block(self) -> None:
        rng = random.Random(9)
        t = _t(["....", "....", "...."])
        nbrs = t.neighbor_map()
        for _ in range(50):
            P = _random_partition(t, rng.randint(1, 6), rng)
            p = P.largest_block()
            B = P.blocks[p]
            if len(B) < 2:
                continue
            half = bfs_bisect(B, t)
            assert 0 < len(half) == len(B) // 2 and half < B
            Q = refine_partition(P, half)
            assert len(Q) == len(P) + 1
            if (
                len(_components(tuple(0 if i in B else 1 for i in range(12)), 0, nbrs))
                == 1
            ):
                owner = tuple(0 if i in half else 1 for i in range(12))
                assert len(_components(owner, 0, nbrs)) == 1

    def test_bfs_bisect_small_blocks(self, territory_2x2: Territory) -> None:
        assert bfs_bisect({2}, territory_2x2) == frozenset()
        assert bfs_bisect(set(), territory_2x2) == frozenset()


def test_abstract_rcmas_is_frozen(territory_2x2: Territory) -> None:
    A = build_abstract_rcmas(territory_2x2, 2, trivial_partition(territory_2x2))
    assert isinstance(A, AbstractRCMAS)
    with pytest.raises(Exception):
        A.horizon = 3  # type: ignore[misc]
