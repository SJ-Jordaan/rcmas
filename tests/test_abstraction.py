"""Tests for rcmas.abstraction (Defs 8-12)."""

from __future__ import annotations

import pytest

from rcmas.abstraction import (
    AbstractRCMAS,
    LiftedStrategy,
    Partition,
    build_abstract_rcmas,
    compute_deviation_set,
    discrete_partition,
    lift_strategy,
    orbit_partition,
    refine_partition,
)
from rcmas.model import Coord, Territory


# ---------------------------------------------------------------------------
# TestPartition
# ---------------------------------------------------------------------------

class TestPartition:
    def test_discrete_partition_2x2(self, territory_2x2: Territory) -> None:
        p = discrete_partition(territory_2x2)
        assert len(p.blocks) == 4
        assert all(len(b) == 1 for b in p.blocks)
        # Each sector maps to its own block
        for i in range(4):
            assert p.membership[i] == i

    def test_discrete_partition_membership_coverage(self, territory_3x3: Territory) -> None:
        p = discrete_partition(territory_3x3)
        assert len(p.blocks) == 9
        all_sectors = set()
        for b in p.blocks:
            all_sectors |= b
        assert all_sectors == set(range(9))

    def test_orbit_partition_2x2(self, territory_2x2: Territory) -> None:
        """2x2 square is fully symmetric: all 4 sectors in one orbit."""
        p = orbit_partition(territory_2x2)
        assert len(p.blocks) == 1
        assert p.blocks[0] == frozenset({0, 1, 2, 3})

    def test_orbit_partition_3x3(self, territory_3x3: Territory) -> None:
        """3x3 square has 3 orbits: 4 corners, 4 edges, 1 centre."""
        p = orbit_partition(territory_3x3)
        assert len(p.blocks) == 3
        sizes = sorted(len(b) for b in p.blocks)
        assert sizes == [1, 4, 4]

    def test_block_disjointness(self, territory_3x3: Territory) -> None:
        p = orbit_partition(territory_3x3)
        all_sectors: set[int] = set()
        for b in p.blocks:
            assert all_sectors.isdisjoint(b), "Blocks overlap"
            all_sectors |= b
        assert all_sectors == set(range(9))

    def test_discrete_partition_L_shape(self, territory_L_shape: Territory) -> None:
        p = discrete_partition(territory_L_shape)
        assert len(p.blocks) == 5
        for i in range(5):
            assert frozenset({i}) in p.blocks


# ---------------------------------------------------------------------------
# TestAbstractRCMAS
# ---------------------------------------------------------------------------

class TestAbstractRCMAS:
    def test_weights_equal_block_sizes_2x2(self, territory_2x2: Territory) -> None:
        p = discrete_partition(territory_2x2)
        ab = build_abstract_rcmas(territory_2x2, num_agents=2, partition=p)
        assert ab.weights == (1, 1, 1, 1)

    def test_weights_equal_block_sizes_3x3_orbits(self, territory_3x3: Territory) -> None:
        p = orbit_partition(territory_3x3)
        ab = build_abstract_rcmas(territory_3x3, num_agents=3, partition=p)
        # 3 orbits with sizes 1, 4, 4 (sorted by min element)
        assert sum(ab.weights) == 9
        sizes = sorted(ab.weights)
        assert sizes == [1, 4, 4]

    def test_horizon_k_over_n(self, territory_3x3: Territory) -> None:
        p = orbit_partition(territory_3x3)
        # 3 blocks, 3 agents -> horizon = 1
        ab = build_abstract_rcmas(territory_3x3, num_agents=3, partition=p)
        assert ab.horizon == len(p.blocks) // 3

    def test_abstract_adjacency_2x2_discrete(self, territory_2x2: Territory) -> None:
        """Discrete partition on 2x2 should preserve concrete adjacency."""
        p = discrete_partition(territory_2x2)
        ab = build_abstract_rcmas(territory_2x2, num_agents=2, partition=p)
        # 2x2 grid: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
        # 0~1, 0~2, 1~3, 2~3
        for i in range(4):
            assert i in ab.neighbors
        assert 1 in ab.neighbors[0]
        assert 2 in ab.neighbors[0]
        assert 3 in ab.neighbors[1]
        assert 3 in ab.neighbors[2]

    def test_abstract_adjacency_orbits(self, territory_3x3: Territory) -> None:
        """Orbit partition on 3x3: centre~edges and edges~corners, but not centre~corners."""
        p = orbit_partition(territory_3x3)
        ab = build_abstract_rcmas(territory_3x3, num_agents=3, partition=p)
        # 3x3 orbits (sorted by min): {4}=centre, {0,2,6,8}=corners, {1,3,5,7}=edges
        # Find which block index holds the centre (sector 4)
        centre_block = p.membership[4]
        corner_block = p.membership[0]
        edge_block = p.membership[1]
        # Centre is adjacent to edges, edges adjacent to corners
        assert edge_block in ab.neighbors[centre_block]
        assert corner_block in ab.neighbors[edge_block]
        # Centre is NOT adjacent to corners
        assert corner_block not in ab.neighbors[centre_block]

    def test_synthetic_territory_size(self, territory_3x3: Territory) -> None:
        p = orbit_partition(territory_3x3)
        ab = build_abstract_rcmas(territory_3x3, num_agents=2, partition=p)
        assert len(ab.territory) == len(p.blocks)

    def test_concrete_territory_preserved(self, territory_2x2: Territory) -> None:
        p = discrete_partition(territory_2x2)
        ab = build_abstract_rcmas(territory_2x2, num_agents=2, partition=p)
        assert ab.concrete_territory is territory_2x2


# ---------------------------------------------------------------------------
# TestLiftStrategy — requires Z3
# ---------------------------------------------------------------------------

z3 = pytest.importorskip("z3")


class TestLiftStrategy:
    def test_block_ownership_inheritance(self, territory_2x2: Territory) -> None:
        """Lifting should map abstract block ownership to all concrete sectors in that block."""
        # Create orbit partition: 2x2 has 1 block of 4 sectors
        p = orbit_partition(territory_2x2)
        ab = build_abstract_rcmas(territory_2x2, num_agents=1, partition=p)

        # Solve the abstract game
        from rcmas.ibis import solve_ibis

        result = solve_ibis(
            territory=ab.territory,
            num_agents=1,
            horizon=ab.horizon,
            weights=ab.weights,
            custom_neighbors=ab.neighbors,
        )
        assert result.is_sat
        assert result.final_solution is not None

        lifted = lift_strategy(ab, result.final_solution)
        # All concrete sectors should be owned by the same agent
        assert all(o == lifted.terminal_owner[0] for o in lifted.terminal_owner)

    def test_concrete_rewards_computed(self, territory_2x1: Territory) -> None:
        """Concrete rewards should be computed using largest_region_size."""
        p = discrete_partition(territory_2x1)
        ab = build_abstract_rcmas(territory_2x1, num_agents=2, partition=p)

        from rcmas.ibis import solve_ibis

        result = solve_ibis(
            territory=ab.territory,
            num_agents=2,
            horizon=ab.horizon,
            weights=ab.weights,
            custom_neighbors=ab.neighbors,
        )
        assert result.is_sat
        assert result.final_solution is not None

        lifted = lift_strategy(ab, result.final_solution)
        assert len(lifted.payoff_by_agent) == 2
        assert all(p >= 0 for p in lifted.payoff_by_agent)


# ---------------------------------------------------------------------------
# TestDeviationSet
# ---------------------------------------------------------------------------

class TestDeviationSet:
    def test_identical_terminals(self) -> None:
        orig = (0, 1, 0, 1)
        assert compute_deviation_set(orig, orig) == frozenset()

    def test_all_different(self) -> None:
        orig = (0, 0, 0, 0)
        dev = (1, 1, 1, 1)
        assert compute_deviation_set(orig, dev) == frozenset({0, 1, 2, 3})

    def test_partial_difference(self) -> None:
        orig = (0, 1, 0, 1)
        dev = (0, 0, 0, 1)
        assert compute_deviation_set(orig, dev) == frozenset({1})


# ---------------------------------------------------------------------------
# TestRefinePartition
# ---------------------------------------------------------------------------

class TestRefinePartition:
    def test_split_single_block(self) -> None:
        p = Partition(
            blocks=(frozenset({0, 1, 2, 3}),),
            membership=(0, 0, 0, 0),
        )
        delta = frozenset({1, 2})
        refined = refine_partition(p, delta)
        assert len(refined.blocks) == 2
        assert frozenset({1, 2}) in refined.blocks
        assert frozenset({0, 3}) in refined.blocks

    def test_no_split_when_all_inside(self) -> None:
        p = Partition(
            blocks=(frozenset({0, 1}), frozenset({2, 3})),
            membership=(0, 0, 1, 1),
        )
        delta = frozenset({0, 1})
        refined = refine_partition(p, delta)
        assert len(refined.blocks) == 2  # No split since {0,1} ⊂ block0 entirely

    def test_count_increases(self) -> None:
        p = Partition(
            blocks=(frozenset({0, 1, 2}), frozenset({3, 4, 5})),
            membership=(0, 0, 0, 1, 1, 1),
        )
        delta = frozenset({1, 4})
        refined = refine_partition(p, delta)
        assert len(refined.blocks) == 4  # Both blocks split

    def test_preserves_unsplit_blocks(self) -> None:
        p = Partition(
            blocks=(frozenset({0, 1}), frozenset({2, 3})),
            membership=(0, 0, 1, 1),
        )
        delta = frozenset({0})
        refined = refine_partition(p, delta)
        # Only first block splits
        assert len(refined.blocks) == 3
        assert frozenset({2, 3}) in refined.blocks

    def test_membership_correct_after_split(self) -> None:
        p = Partition(
            blocks=(frozenset({0, 1, 2, 3}),),
            membership=(0, 0, 0, 0),
        )
        delta = frozenset({1, 3})
        refined = refine_partition(p, delta)
        # Sectors 0,2 should be in one block, 1,3 in another
        assert refined.membership[0] == refined.membership[2]
        assert refined.membership[1] == refined.membership[3]
        assert refined.membership[0] != refined.membership[1]
