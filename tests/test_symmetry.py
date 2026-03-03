"""Tests for the symmetry module and symmetry-breaking integration."""

from __future__ import annotations

import pytest

from rcmas.model import Coord, Territory
from rcmas.symmetry import (
    SymmetryInfo,
    canonical_state,
    demand_classes,
    invert_automorphism,
    orbit_representatives,
    sector_orbits,
    symmetry_info,
    territory_automorphisms,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def territory_2x2() -> Territory:
    return Territory.from_ascii(["..", ".."])


@pytest.fixture
def territory_3x3() -> Territory:
    return Territory.from_ascii(["...", "...", "..."])


@pytest.fixture
def territory_2x3() -> Territory:
    return Territory.from_ascii(["...", "..."])


@pytest.fixture
def territory_L_shape() -> Territory:
    return Territory.from_ascii(["..", ".", ".."])


@pytest.fixture
def territory_1x1() -> Territory:
    return Territory.from_ascii(["."])


# ---------------------------------------------------------------------------
# Automorphism detection tests
# ---------------------------------------------------------------------------

class TestTerritoryAutomorphisms:

    def test_2x2_square_has_8_automorphisms(self, territory_2x2: Territory):
        """2x2 square has full D_4 symmetry: 8 automorphisms."""
        auts = territory_automorphisms(territory_2x2)
        assert len(auts) == 8

    def test_3x3_square_has_8_automorphisms(self, territory_3x3: Territory):
        """3x3 square also has D_4 symmetry: 8 automorphisms."""
        auts = territory_automorphisms(territory_3x3)
        assert len(auts) == 8

    def test_2x3_rectangle_has_4_automorphisms(self, territory_2x3: Territory):
        """2x3 rectangle has V_4 (Klein four-group): 4 automorphisms."""
        auts = territory_automorphisms(territory_2x3)
        assert len(auts) == 4

    def test_L_shape_has_2_automorphisms(self, territory_L_shape: Territory):
        """L-shaped territory has identity + vertical flip (2 automorphisms)."""
        auts = territory_automorphisms(territory_L_shape)
        assert len(auts) == 2

    def test_1x1_has_1_automorphism(self, territory_1x1: Territory):
        """Single-cell territory has only the identity (all D_4 maps collapse to it)."""
        auts = territory_automorphisms(territory_1x1)
        assert len(auts) == 1

    def test_identity_is_first(self, territory_2x2: Territory):
        """Identity permutation should always be the first automorphism."""
        auts = territory_automorphisms(territory_2x2)
        S = len(territory_2x2)
        identity = {i: i for i in range(S)}
        assert auts[0] == identity

    def test_automorphisms_are_permutations(self, territory_3x3: Territory):
        """Each automorphism must be a bijection on {0, ..., S-1}."""
        auts = territory_automorphisms(territory_3x3)
        S = len(territory_3x3)
        for perm in auts:
            assert set(perm.keys()) == set(range(S))
            assert set(perm.values()) == set(range(S))

    def test_empty_territory(self):
        """Empty territory has trivial automorphism group."""
        t = Territory.from_ascii(["###"])
        auts = territory_automorphisms(t)
        assert len(auts) == 1
        assert auts[0] == {}


# ---------------------------------------------------------------------------
# Orbit computation tests
# ---------------------------------------------------------------------------

class TestSectorOrbits:

    def test_2x2_orbits(self, territory_2x2: Territory):
        """2x2 square with D_4: all 4 sectors are in one orbit."""
        auts = territory_automorphisms(territory_2x2)
        orbs = sector_orbits(territory_2x2, auts)
        assert len(orbs) == 1
        assert orbs[0] == frozenset({0, 1, 2, 3})

    def test_3x3_orbits(self, territory_3x3: Territory):
        """3x3 square with D_4: corners form one orbit, edges another, centre a third."""
        auts = territory_automorphisms(territory_3x3)
        orbs = sector_orbits(territory_3x3, auts)
        # 3x3 has 3 orbits: 4 corners, 4 edges, 1 centre
        assert len(orbs) == 3
        orbit_sizes = sorted(len(o) for o in orbs)
        assert orbit_sizes == [1, 4, 4]

    def test_2x3_orbits(self, territory_2x3: Territory):
        """2x3 rectangle with V_4: corner orbit (4) + edge orbit (2)."""
        auts = territory_automorphisms(territory_2x3)
        orbs = sector_orbits(territory_2x3, auts)
        assert len(orbs) == 2
        orbit_sizes = sorted(len(o) for o in orbs)
        assert orbit_sizes == [2, 4]

    def test_L_shape_orbits(self, territory_L_shape: Territory):
        """L-shape has vertical flip symmetry: top-bottom pairs share orbits."""
        auts = territory_automorphisms(territory_L_shape)
        orbs = sector_orbits(territory_L_shape, auts)
        # 5 sectors, vertical flip pairs (0,3) and (1,4), singleton (2) -> 3 orbits
        assert len(orbs) == 3

    def test_orbit_cover(self, territory_3x3: Territory):
        """All sector indices must appear in exactly one orbit."""
        auts = territory_automorphisms(territory_3x3)
        orbs = sector_orbits(territory_3x3, auts)
        all_indices = set()
        for orb in orbs:
            assert orb.isdisjoint(all_indices)
            all_indices.update(orb)
        assert all_indices == set(range(len(territory_3x3)))


# ---------------------------------------------------------------------------
# Orbit representatives
# ---------------------------------------------------------------------------

class TestOrbitRepresentatives:

    def test_representatives_are_minima(self, territory_3x3: Territory):
        """Each representative should be the minimum index of its orbit."""
        auts = territory_automorphisms(territory_3x3)
        orbs = sector_orbits(territory_3x3, auts)
        reps = orbit_representatives(orbs)
        assert len(reps) == len(orbs)
        for rep, orb in zip(sorted(reps), sorted(orbs, key=min)):
            assert rep == min(orb)

    def test_2x2_single_representative(self, territory_2x2: Territory):
        """2x2 square: one orbit -> one representative (index 0)."""
        auts = territory_automorphisms(territory_2x2)
        orbs = sector_orbits(territory_2x2, auts)
        reps = orbit_representatives(orbs)
        assert reps == [0]


# ---------------------------------------------------------------------------
# SymmetryInfo convenience function
# ---------------------------------------------------------------------------

class TestSymmetryInfo:

    def test_symmetry_info_bundles(self, territory_3x3: Territory):
        """symmetry_info returns consistent SymmetryInfo."""
        si = symmetry_info(territory_3x3)
        assert isinstance(si, SymmetryInfo)
        assert len(si.automorphisms) == 8
        assert len(si.orbits) == 3
        assert len(si.representatives) == 3


# ---------------------------------------------------------------------------
# Integration: symmetry-breaking with CO solver
# ---------------------------------------------------------------------------

class TestSymmetryBreakingIntegration:

    def test_co_with_symmetry_2x2(self, territory_2x2: Territory):
        """CO solve with symmetry breaking on 2x2 still produces a valid solution."""
        from rcmas.smt_solve import solve_collective_optimality

        sol = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True,
        )
        assert sol.is_sat
        assert sol.reason == "sat"
        assert sol.final_state is not None

    def test_co_with_symmetry_3x3(self, territory_3x3: Territory):
        """CO solve with symmetry breaking on 3x3 still produces a valid solution."""
        from rcmas.smt_solve import solve_collective_optimality

        # 3x3 has 9 sectors; 2 agents need horizon=4 (8 claims, 1 unclaimed) to avoid
        # the final-round deadlock where only 1 sector remains for 2 agents.
        sol = solve_collective_optimality(
            territory=territory_3x3, num_agents=2, horizon=4,
            debug=True, symmetry_breaking=True,
        )
        assert sol.is_sat
        assert sol.final_state is not None

    def test_co_payoff_preserved(self, territory_2x2: Territory):
        """Symmetry breaking should not reduce the optimal payoff."""
        from rcmas.smt_solve import solve_collective_optimality

        sol_no_sym = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=False,
        )
        sol_sym = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True,
        )
        assert sol_no_sym.is_sat and sol_sym.is_sat
        assert sol_no_sym.payoff_by_agent is not None
        assert sol_sym.payoff_by_agent is not None
        assert sum(sol_sym.payoff_by_agent) == sum(sol_no_sym.payoff_by_agent)

    def test_smt_solve_with_symmetry(self, territory_2x2: Territory):
        """Direct solve_smt_game with symmetry_breaking=True."""
        from rcmas.smt_solve import solve_smt_game

        sol = solve_smt_game(
            territory=territory_2x2, num_agents=2, horizon=2,
            objective="sum", debug=True, symmetry_breaking=True,
        )
        assert sol.is_sat

    def test_ibis_with_symmetry(self, territory_2x2: Territory):
        """IBIS with symmetry on a small grid finds NE or terminates cleanly."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_2x2, num_agents=2, horizon=2,
            max_iters=10, symmetry=True,
        )
        assert res.is_sat

    def test_symmetry_on_asymmetric_territory(self, territory_L_shape: Territory):
        """Symmetry breaking on an L-shaped territory should still work."""
        from rcmas.smt_solve import solve_collective_optimality

        # L-shape has 5 sectors; 2 agents × 2 rounds = 4 claims, leaving 1 unclaimed.
        sol = solve_collective_optimality(
            territory=territory_L_shape, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True,
        )
        assert sol.is_sat


# ---------------------------------------------------------------------------
# CLI --symmetry flag
# ---------------------------------------------------------------------------

class TestCliSymmetryFlag:

    def test_parse_co_symmetry(self):
        from rcmas.cli import _parse_args

        args = _parse_args(["co", "--grid", "grids/symmetric/2x2.txt", "--agents", "2", "--horizon", "2", "--symmetry"])
        assert args.symmetry is True

    def test_parse_ibis_symmetry(self):
        from rcmas.cli import _parse_args

        args = _parse_args(["ibis", "--grid", "grids/symmetric/2x2.txt", "--agents", "2", "--horizon", "2", "--symmetry"])
        assert args.symmetry is True

    def test_parse_qibis_symmetry(self):
        from rcmas.cli import _parse_args

        args = _parse_args(["qibis", "--grid", "grids/symmetric/2x2.txt", "--agents", "2", "--horizon", "2", "--symmetry"])
        assert args.symmetry is True

    def test_parse_default_no_symmetry(self):
        from rcmas.cli import _parse_args

        args = _parse_args(["co", "--grid", "grids/symmetric/2x2.txt", "--agents", "2", "--horizon", "2"])
        assert args.symmetry is False


# ---------------------------------------------------------------------------
# Canonical state mapping
# ---------------------------------------------------------------------------

class TestCanonicalState:

    def test_canonical_state_identity(self, territory_1x1: Territory):
        """Single-cell territory: canonical state is the state itself."""
        si = symmetry_info(territory_1x1)
        owner = (0,)
        canon, sigma = canonical_state(owner, si.automorphisms)
        assert canon == owner
        assert sigma == {0: 0}

    def test_canonical_state_2x2(self, territory_2x2: Territory):
        """Two symmetric states on a 2x2 grid canonicalize to the same tuple."""
        si = symmetry_info(territory_2x2)
        # Agent 0 owns sector 0, rest unowned
        state_a = (0, -1, -1, -1)
        # Agent 0 owns sector 3 (opposite corner), rest unowned
        state_b = (-1, -1, -1, 0)
        canon_a, _ = canonical_state(state_a, si.automorphisms)
        canon_b, _ = canonical_state(state_b, si.automorphisms)
        assert canon_a == canon_b

    def test_canonical_state_is_lexmin(self, territory_2x2: Territory):
        """Canonical state is the lex-minimum over all automorphism images."""
        si = symmetry_info(territory_2x2)
        owner = (-1, 0, -1, -1)
        canon, _sigma = canonical_state(owner, si.automorphisms)
        S = len(owner)
        for aut in si.automorphisms:
            transformed = [0] * S
            for i in range(S):
                transformed[aut[i]] = owner[i]
            assert canon <= tuple(transformed)

    def test_canonical_state_returns_valid_sigma(self, territory_2x2: Territory):
        """The returned sigma is an automorphism and produces the canonical tuple."""
        si = symmetry_info(territory_2x2)
        owner = (0, -1, 1, -1)
        canon, sigma = canonical_state(owner, si.automorphisms)
        assert sigma in si.automorphisms
        S = len(owner)
        transformed = [0] * S
        for i in range(S):
            transformed[sigma[i]] = owner[i]
        assert tuple(transformed) == canon

    def test_invert_round_trips(self, territory_2x2: Territory):
        """inv(sigma) . sigma = id for all automorphisms."""
        si = symmetry_info(territory_2x2)
        for sigma in si.automorphisms:
            inv = invert_automorphism(sigma)
            S = len(sigma)
            for i in range(S):
                assert inv[sigma[i]] == i


# ---------------------------------------------------------------------------
# Demand classes
# ---------------------------------------------------------------------------

class TestDemandClasses:

    def test_uniform_demands_single_class(self):
        """All agents with the same demand form one class."""
        classes = demand_classes((3, 3, 3))
        assert len(classes) == 1
        assert classes[0] == [0, 1, 2]

    def test_two_groups(self):
        """Agents with two distinct demands produce two classes."""
        classes = demand_classes((2, 2, 5, 5))
        assert len(classes) == 2
        # Each class should contain agents with the same demand
        class_sets = [frozenset(c) for c in classes]
        assert frozenset({0, 1}) in class_sets
        assert frozenset({2, 3}) in class_sets

    def test_all_different_demands(self):
        """Each agent in its own class when all demands differ."""
        classes = demand_classes((1, 2, 3))
        assert len(classes) == 3
        for cls in classes:
            assert len(cls) == 1

    def test_agents_sorted_within_class(self):
        """Agent indices within each class should be sorted."""
        # Demands: agent 0=5, agent 1=3, agent 2=5, agent 3=3
        classes = demand_classes((5, 3, 5, 3))
        for cls in classes:
            assert cls == sorted(cls)

    def test_empty_demands(self):
        """Empty demands produce no classes."""
        classes = demand_classes(())
        assert classes == []

    def test_single_agent(self):
        """Single agent forms one class."""
        classes = demand_classes((4,))
        assert len(classes) == 1
        assert classes[0] == [0]


# ---------------------------------------------------------------------------
# Demand-class lex-leader constraint
# ---------------------------------------------------------------------------

class TestDemandClassLexLeader:

    def test_uniform_demands_same_as_global(self, territory_2x2: Territory):
        """With uniform demands, behaviour matches the global lex-leader."""
        from rcmas.smt_solve import solve_collective_optimality

        sol_global = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True,
        )
        sol_demand = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True, demands=(2, 2),
        )
        assert sol_global.is_sat and sol_demand.is_sat
        assert sol_global.payoff_by_agent is not None
        assert sol_demand.payoff_by_agent is not None
        assert sum(sol_global.payoff_by_agent) == sum(sol_demand.payoff_by_agent)

    def test_heterogeneous_demands_still_sat(self, territory_2x2: Territory):
        """Non-uniform demands with symmetry breaking still finds a solution."""
        from rcmas.smt_solve import solve_collective_optimality

        sol = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True, demands=(1, 2),
        )
        assert sol.is_sat

    def test_demands_none_is_global_lex_leader(self, territory_2x2: Territory):
        """demands=None should use global lex-leader (backward compatible)."""
        from rcmas.smt_solve import solve_smt_game

        sol = solve_smt_game(
            territory=territory_2x2, num_agents=2, horizon=2,
            objective="sum", debug=True, symmetry_breaking=True,
            demands=None,
        )
        assert sol.is_sat

    def test_all_different_demands_no_agent_constraint(self, territory_3x3: Territory):
        """All-different demands: no lex-leader constraints, only spatial."""
        from rcmas.smt_solve import solve_collective_optimality

        # 3 agents on 3x3, all different demands — no agent-permutation
        # symmetry, but spatial canonicalisation still applies.
        sol = solve_collective_optimality(
            territory=territory_3x3, num_agents=3, horizon=3,
            debug=True, symmetry_breaking=True, demands=(1, 2, 3),
        )
        assert sol.is_sat

    def test_ibis_with_demands(self, territory_2x2: Territory):
        """IBIS with demand classes finds NE or terminates cleanly."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_2x2, num_agents=2, horizon=2,
            max_iters=10, symmetry=True, demands=(2, 2),
        )
        assert res.is_sat
