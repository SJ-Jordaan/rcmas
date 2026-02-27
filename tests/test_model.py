"""Tests for model.py (Def 1-8)."""

from __future__ import annotations

import io

import pytest

from rcmas.model import (
    UNOWNED,
    CollisionError,
    Coord,
    State,
    Territory,
    adjacent_owned,
    connected_components,
    evolve,
    largest_region_size,
    neighbors4,
    scores,
)


# ===================================================================
# Def 1: Coord
# ===================================================================

class TestCoord:
    def test_frozen(self):
        c = Coord(1, 2)
        with pytest.raises(AttributeError):
            c.x = 3  # type: ignore[misc]

    def test_equality(self):
        assert Coord(0, 0) == Coord(0, 0)
        assert Coord(0, 0) != Coord(1, 0)

    def test_hashable(self):
        s = {Coord(0, 0), Coord(0, 0), Coord(1, 0)}
        assert len(s) == 2

    def test_ordering_independent_of_creation(self):
        """Coords with same values are always equal regardless of how they're created."""
        a = Coord(x=3, y=4)
        b = Coord(3, 4)
        assert a == b
        assert hash(a) == hash(b)


# ===================================================================
# Def 2: 4-connected neighbourhood
# ===================================================================

class TestNeighbors4:
    def test_yields_four_cardinal(self):
        nbs = list(neighbors4(Coord(1, 1)))
        assert len(nbs) == 4
        assert set(nbs) == {Coord(2, 1), Coord(0, 1), Coord(1, 2), Coord(1, 0)}

    def test_at_origin(self):
        nbs = list(neighbors4(Coord(0, 0)))
        assert Coord(-1, 0) in nbs  # neighbours can be negative (not in territory)
        assert Coord(0, -1) in nbs

    def test_does_not_include_diagonals(self):
        nbs = set(neighbors4(Coord(1, 1)))
        assert Coord(0, 0) not in nbs
        assert Coord(2, 2) not in nbs


# ===================================================================
# Def 3: Territory
# ===================================================================

class TestTerritory:
    def test_from_ascii_2x2(self, territory_2x2):
        assert len(territory_2x2) == 4
        assert Coord(0, 0) in territory_2x2.sectors
        assert Coord(1, 1) in territory_2x2.sectors

    def test_from_ascii_ignores_non_sector_chars(self):
        t = Territory.from_ascii(["..#", ".#.", "#.."])
        assert len(t) == 6
        assert t.index_of(Coord(2, 0)) is None
        assert t.index_of(Coord(1, 1)) is None
        assert t.index_of(Coord(0, 2)) is None

    def test_from_ascii_custom_sector_char(self):
        t = Territory.from_ascii(["xx.x", "x..x"], sector="x")
        assert len(t) == 5

    def test_index_of_valid(self, territory_2x2):
        idx = territory_2x2.index_of(Coord(0, 0))
        assert idx is not None
        assert 0 <= idx < 4

    def test_index_of_outside_territory(self, territory_2x2):
        assert territory_2x2.index_of(Coord(99, 99)) is None

    def test_ordered_sectors_row_major(self, territory_2x2):
        """Sectors are ordered by (y, x) — row-major."""
        sectors = territory_2x2.ordered_sectors()
        assert sectors == (Coord(0, 0), Coord(1, 0), Coord(0, 1), Coord(1, 1))

    def test_ordered_sectors_are_all_sector_coords(self, territory_3x3):
        sectors = territory_3x3.ordered_sectors()
        assert set(sectors) == territory_3x3.sectors

    def test_index_of_is_inverse_of_ordered(self, territory_3x3):
        sectors = territory_3x3.ordered_sectors()
        for i, c in enumerate(sectors):
            assert territory_3x3.index_of(c) == i

    def test_from_ascii_handles_io_stringio(self):
        t = Territory.from_ascii(io.StringIO("...\n...\n"))
        assert len(t) == 6

    def test_empty_territory(self):
        t = Territory.from_ascii(["###"])
        assert len(t) == 0

    def test_single_sector(self, territory_1x1):
        assert len(territory_1x1) == 1
        assert territory_1x1.index_of(Coord(0, 0)) == 0

    def test_disconnected_territory(self, territory_disconnected):
        """Sectors at (0,0),(1,0) and (3,0),(4,0) separated by '#' at (2,0)."""
        assert len(territory_disconnected) == 4
        assert territory_disconnected.index_of(Coord(2, 0)) is None


# ===================================================================
# Def 4-5: State + ownership
# ===================================================================

class TestState:
    def test_initial_all_unowned(self, state_2x2):
        assert all(o == UNOWNED for o in state_2x2.owner_by_index)
        assert state_2x2.round_index == 0

    def test_initial_rejects_zero_agents(self, territory_2x2):
        with pytest.raises(ValueError):
            State.initial(territory_2x2, num_agents=0)

    def test_initial_rejects_negative_agents(self, territory_2x2):
        with pytest.raises(ValueError):
            State.initial(territory_2x2, num_agents=-1)

    def test_available_actions_all_at_start(self, state_2x2):
        assert len(state_2x2.available_actions()) == 4

    def test_available_actions_shrinks_after_claim(self, state_2x2):
        s1 = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        assert len(s1.available_actions()) == 2

    def test_available_actions_empty_at_terminal(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        s = evolve(s, {0: Coord(0, 1), 1: Coord(1, 1)})
        assert len(s.available_actions()) == 0

    def test_is_terminal_false_at_start(self, state_2x2):
        assert not state_2x2.is_terminal()

    def test_is_terminal_true_when_full(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        s = evolve(s, {0: Coord(0, 1), 1: Coord(1, 1)})
        assert s.is_terminal()

    def test_owner_of_unowned_returns_none(self, state_2x2):
        assert state_2x2.owner_of(Coord(0, 0)) is None

    def test_owner_of_outside_territory_returns_none(self, state_2x2):
        assert state_2x2.owner_of(Coord(99, 99)) is None

    def test_owner_of_claimed_sector(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0)})
        assert s.owner_of(Coord(0, 0)) == 0

    def test_state_is_frozen(self, state_2x2):
        with pytest.raises(AttributeError):
            state_2x2.round_index = 5  # type: ignore[misc]

    def test_num_agents_preserved(self, territory_3x3):
        s = State.initial(territory_3x3, num_agents=3)
        assert s.num_agents == 3


# ===================================================================
# Def 6: evolve
# ===================================================================

class TestEvolve:
    def test_claims_sector(self, state_2x2):
        s1 = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        assert s1.owner_of(Coord(0, 0)) == 0
        assert s1.owner_of(Coord(1, 0)) == 1
        assert s1.round_index == 1

    def test_collision_raises(self, state_2x2):
        with pytest.raises(CollisionError):
            evolve(state_2x2, {0: Coord(0, 0), 1: Coord(0, 0)})

    def test_none_action_is_pass(self, state_2x2):
        s1 = evolve(state_2x2, {0: Coord(0, 0), 1: None})
        assert s1.owner_of(Coord(0, 0)) == 0
        assert s1.owner_of(Coord(1, 0)) is None
        assert s1.round_index == 1

    def test_invalid_agent_raises(self, state_2x2):
        with pytest.raises(ValueError, match="invalid agent_id"):
            evolve(state_2x2, {5: Coord(0, 0)})

    def test_owned_sector_raises(self, state_2x2):
        s1 = evolve(state_2x2, {0: Coord(0, 0)})
        with pytest.raises(ValueError, match="owned sector"):
            evolve(s1, {0: Coord(0, 0)})

    def test_out_of_territory_raises(self, state_2x2):
        with pytest.raises(ValueError, match="not in territory"):
            evolve(state_2x2, {0: Coord(99, 99)})

    def test_terminal_is_identity(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        s = evolve(s, {0: Coord(0, 1), 1: Coord(1, 1)})
        assert s.is_terminal()
        s2 = evolve(s, {0: Coord(0, 0)})
        assert s2 is s  # identity, not just equality

    def test_round_index_increments(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0)})
        assert s.round_index == 1
        s = evolve(s, {0: Coord(1, 0)})
        assert s.round_index == 2

    def test_empty_actions_is_all_pass(self, state_2x2):
        s = evolve(state_2x2, {})
        assert s.round_index == 1
        assert all(o == UNOWNED for o in s.owner_by_index)

    def test_three_agent_game(self, territory_3x3):
        s = State.initial(territory_3x3, num_agents=3)
        s = evolve(s, {0: Coord(0, 0), 1: Coord(1, 0), 2: Coord(2, 0)})
        assert s.owner_of(Coord(0, 0)) == 0
        assert s.owner_of(Coord(1, 0)) == 1
        assert s.owner_of(Coord(2, 0)) == 2

    def test_three_way_collision_raises(self, territory_3x3):
        s = State.initial(territory_3x3, num_agents=3)
        with pytest.raises(CollisionError):
            evolve(s, {0: Coord(0, 0), 1: Coord(0, 0), 2: Coord(0, 0)})

    def test_ownership_persists_across_rounds(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0)})
        s = evolve(s, {1: Coord(1, 0)})
        assert s.owner_of(Coord(0, 0)) == 0  # still owned by 0


# ===================================================================
# Def 7: Adjacency + connected components
# ===================================================================

class TestAdjacency:
    def test_adjacent_owned_empty_at_start(self, state_2x2):
        assert adjacent_owned(state_2x2, 0) == set()

    def test_adjacent_owned_horizontal_pair(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 0)})
        pairs = adjacent_owned(s, 0)
        # (0,0) idx=0 and (1,0) idx=1 are adjacent
        assert len(pairs) == 1
        assert (0, 1) in pairs

    def test_adjacent_owned_excludes_other_agent(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        assert adjacent_owned(s, 0) == set()  # agent 1 owns the neighbour
        assert adjacent_owned(s, 1) == set()

    def test_adjacent_owned_diagonal_not_adjacent(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 1)})
        assert adjacent_owned(s, 0) == set()


class TestConnectedComponents:
    def test_no_ownership(self, state_2x2):
        assert connected_components(state_2x2, 0) == []

    def test_single_sector(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0)})
        comps = connected_components(s, 0)
        assert len(comps) == 1
        assert comps[0] == {Coord(0, 0)}

    def test_two_disconnected_diagonal(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 1)})
        comps = connected_components(s, 0)
        assert len(comps) == 2

    def test_merge_via_horizontal_neighbour(self, state_2x2):
        s = evolve(state_2x2, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 0)})
        comps = connected_components(s, 0)
        assert len(comps) == 1
        assert comps[0] == {Coord(0, 0), Coord(1, 0)}

    def test_full_board_one_component(self, territory_2x2):
        """When one agent owns the full 2x2 board, there's exactly one component."""
        s = State.initial(territory_2x2, num_agents=1)
        s = evolve(s, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 0)})
        s = evolve(s, {0: Coord(0, 1)})
        s = evolve(s, {0: Coord(1, 1)})
        comps = connected_components(s, 0)
        assert len(comps) == 1
        assert len(comps[0]) == 4

    def test_l_shape_single_component(self, territory_L_shape):
        """An L-shape owned by one agent should form one connected component."""
        s = State.initial(territory_L_shape, num_agents=1)
        for coord in territory_L_shape.ordered_sectors():
            s = evolve(s, {0: coord})
        comps = connected_components(s, 0)
        assert len(comps) == 1
        assert len(comps[0]) == 5

    def test_disconnected_territory_two_components(self, territory_disconnected):
        """On ..#.. territory, one agent owning all 4 sectors gets 2 components."""
        s = State.initial(territory_disconnected, num_agents=1)
        for coord in territory_disconnected.ordered_sectors():
            s = evolve(s, {0: coord})
        comps = connected_components(s, 0)
        assert len(comps) == 2


# ===================================================================
# Def 8: Largest region size / scores
# ===================================================================

class TestLargestRegionAndScores:
    def test_zero_when_nothing_owned(self, state_2x2):
        assert largest_region_size(state_2x2, 0) == 0

    def test_diagonal_gives_size_one(self):
        t = Territory.from_ascii(["..", ".."])
        s = State.initial(t, num_agents=1)
        s = evolve(s, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 1)})
        assert largest_region_size(s, 0) == 1

    def test_connected_pair(self):
        t = Territory.from_ascii(["..", ".."])
        s = State.initial(t, num_agents=1)
        s = evolve(s, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 0)})
        assert largest_region_size(s, 0) == 2

    def test_bridge_connects_diagonal(self):
        t = Territory.from_ascii(["..", ".."])
        s = State.initial(t, num_agents=1)
        s = evolve(s, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 1)})
        assert largest_region_size(s, 0) == 1
        s = evolve(s, {0: Coord(0, 1)})
        assert largest_region_size(s, 0) == 3

    def test_scores_symmetric_split(self, territory_2x2):
        s = State.initial(territory_2x2, num_agents=2)
        s = evolve(s, {0: Coord(0, 0), 1: Coord(1, 0)})
        s = evolve(s, {0: Coord(0, 1), 1: Coord(1, 1)})
        assert s.is_terminal()
        assert scores(s) == (2, 2)

    def test_scores_asymmetric_split(self, territory_3x3):
        """On 3x3 with 2 agents: one gets 5 connected, one gets 4 connected."""
        s = State.initial(territory_3x3, num_agents=2)
        # Agent 0 takes entire left column + top-middle
        s = evolve(s, {0: Coord(0, 0), 1: Coord(2, 0)})
        s = evolve(s, {0: Coord(0, 1), 1: Coord(2, 1)})
        s = evolve(s, {0: Coord(0, 2), 1: Coord(2, 2)})
        s = evolve(s, {0: Coord(1, 0), 1: Coord(1, 2)})
        # Agent 0 owns: (0,0),(0,1),(0,2),(1,0) = 4 connected
        # Agent 1 owns: (2,0),(2,1),(2,2),(1,2) = 4 connected
        # (1,1) is still unowned
        assert largest_region_size(s, 0) == 4
        assert largest_region_size(s, 1) == 4

    def test_disconnected_territory_payoff(self, territory_disconnected):
        """On ..#.. the maximum single-agent region is 2 (one side of the gap)."""
        s = State.initial(territory_disconnected, num_agents=1)
        for coord in territory_disconnected.ordered_sectors():
            s = evolve(s, {0: coord})
        assert largest_region_size(s, 0) == 2

    def test_scores_three_agents(self, territory_3x3):
        s = State.initial(territory_3x3, num_agents=3)
        s = evolve(s, {0: Coord(0, 0), 1: Coord(1, 0), 2: Coord(2, 0)})
        s = evolve(s, {0: Coord(0, 1), 1: Coord(1, 1), 2: Coord(2, 1)})
        s = evolve(s, {0: Coord(0, 2), 1: Coord(1, 2), 2: Coord(2, 2)})
        assert s.is_terminal()
        assert scores(s) == (3, 3, 3)
