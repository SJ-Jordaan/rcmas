"""Tests for local symmetry breaking and disconnected territory handling."""

from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from rcmas.model import Coord, Territory
from rcmas.symmetry import (
    ComponentInfo,
    LocalSymmetryInfo,
    component_info,
    find_maximal_rectangular_subregions,
    local_symmetry_info,
    territory_connected_components,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def territory_2x2() -> Territory:
    return Territory.from_ascii(["..", ".."])


@pytest.fixture
def territory_disconnected_2_2() -> Territory:
    """Two disconnected 2-sector components: ..#.."""
    return Territory.from_ascii(["..#.."])


@pytest.fixture
def territory_disconnected_2x2_2x2() -> Territory:
    """Two disconnected 2x2 squares: ..#..\n..#.."""
    return Territory.from_ascii(["..#..", "..#.."])


@pytest.fixture
def territory_L_shape_large() -> Territory:
    """L-shaped: 4×4 with upper-right 2 sectors removed (14 sectors)."""
    return Territory.from_ascii([
        "..##",
        "....",
        "....",
        "....",
    ])


# ---------------------------------------------------------------------------
# Connected component decomposition
# ---------------------------------------------------------------------------

class TestConnectedComponents:

    def test_connected_territory_single_component(self, territory_2x2: Territory):
        comps = territory_connected_components(territory_2x2)
        assert len(comps) == 1
        assert comps[0] == frozenset({0, 1, 2, 3})

    def test_disconnected_two_components(self, territory_disconnected_2_2: Territory):
        comps = territory_connected_components(territory_disconnected_2_2)
        assert len(comps) == 2
        assert comps[0] == frozenset({0, 1})
        assert comps[1] == frozenset({2, 3})

    def test_disconnected_2x2_2x2(self, territory_disconnected_2x2_2x2: Territory):
        comps = territory_connected_components(territory_disconnected_2x2_2x2)
        assert len(comps) == 2


# ---------------------------------------------------------------------------
# Component isomorphism and ComponentInfo
# ---------------------------------------------------------------------------

class TestComponentInfo:

    def test_connected_territory(self, territory_2x2: Territory):
        ci = component_info(territory_2x2)
        assert len(ci.components) == 1
        assert len(ci.iso_classes) == 1
        assert ci.iso_classes[0] == (0,)
        assert len(ci.isomorphisms) == 0

    def test_disconnected_isomorphic(self, territory_disconnected_2_2: Territory):
        ci = component_info(territory_disconnected_2_2)
        assert len(ci.components) == 2
        assert len(ci.iso_classes) == 1
        assert set(ci.iso_classes[0]) == {0, 1}
        # Isomorphism witness exists
        assert (0, 1) in ci.isomorphisms
        iso = ci.isomorphisms[(0, 1)]
        assert set(iso.keys()) == {0, 1}
        assert set(iso.values()) == {2, 3}

    def test_disconnected_2x2_isomorphic(self, territory_disconnected_2x2_2x2: Territory):
        ci = component_info(territory_disconnected_2x2_2x2)
        assert len(ci.components) == 2
        assert len(ci.iso_classes) == 1
        # Each component should have 8 automorphisms (D4)
        for auts in ci.component_automorphisms:
            assert len(auts) == 8

    def test_disconnected_non_isomorphic(self):
        """3-sector line and 2-sector line are not isomorphic."""
        t = Territory.from_ascii(["...#.."])
        ci = component_info(t)
        assert len(ci.components) == 2
        assert len(ci.iso_classes) == 2  # Two distinct iso classes


# ---------------------------------------------------------------------------
# Local symmetry info
# ---------------------------------------------------------------------------

class TestLocalSymmetryInfo:

    def test_l_shape_4x3_subregion(self, territory_L_shape_large: Territory):
        """The 4×3 bottom strip has V_4 local automorphisms."""
        # Sectors 2-13 form the 4×3 rectangle
        sub = frozenset(range(2, 14))
        lsi = local_symmetry_info(territory_L_shape_large, sub)
        assert len(lsi.automorphisms) == 4  # V_4
        assert len(lsi.orbits) == 4
        assert lsi.boundary == frozenset({2, 3})
        assert len(lsi.boundary_pairs) > 0
        # Representatives should be 4 (one per orbit)
        assert len(lsi.representatives) == 4

    def test_subregion_boundary_computed(self, territory_L_shape_large: Territory):
        sub = frozenset(range(2, 14))
        lsi = local_symmetry_info(territory_L_shape_large, sub)
        # Boundary sectors are those in R adjacent to T\R
        # Sectors 2 and 3 are adjacent to sectors 0 and 1
        assert 2 in lsi.boundary
        assert 3 in lsi.boundary

    def test_empty_subregion_raises(self, territory_2x2: Territory):
        with pytest.raises(ValueError, match="non-empty"):
            local_symmetry_info(territory_2x2, frozenset())

    def test_invalid_subregion_raises(self, territory_2x2: Territory):
        with pytest.raises(ValueError, match="valid sector"):
            local_symmetry_info(territory_2x2, frozenset({99}))


# ---------------------------------------------------------------------------
# Find maximal rectangular subregions
# ---------------------------------------------------------------------------

class TestFindSubregions:

    def test_l_shape_finds_subregion(self, territory_L_shape_large: Territory):
        subs = find_maximal_rectangular_subregions(territory_L_shape_large)
        assert len(subs) >= 1
        # The 4×3 strip (12 sectors) should be found
        sizes = sorted(len(s) for s in subs)
        assert 12 in sizes

    def test_square_no_subregions(self, territory_2x2: Territory):
        """A square grid has no proper subregions (whole grid is handled globally)."""
        subs = find_maximal_rectangular_subregions(territory_2x2)
        assert len(subs) == 0

    def test_connected_rectangle(self):
        """A rectangle has no proper sub-rectangles with non-trivial automorphisms
        that aren't the whole territory (since the whole is handled globally)."""
        t = Territory.from_ascii(["...", "..."])
        subs = find_maximal_rectangular_subregions(t)
        # May find 2x2 sub-rectangles within a 3x2
        for s in subs:
            assert len(s) < len(t)


# ---------------------------------------------------------------------------
# SMT integration: local symmetry
# ---------------------------------------------------------------------------

class TestLocalSymmetrySmtIntegration:

    def test_l_shape_with_symmetry(self, territory_L_shape_large: Territory):
        """L-shaped grid with symmetry breaking should still find a solution."""
        from rcmas.smt_solve import solve_collective_optimality

        sol = solve_collective_optimality(
            territory=territory_L_shape_large,
            num_agents=2,
            horizon=7,
            debug=True,
            symmetry_breaking=True,
        )
        assert sol.is_sat
        assert sol.payoff_by_agent is not None
        assert sum(sol.payoff_by_agent) > 0

    def test_l_shape_payoff_preserved(self, territory_L_shape_large: Territory):
        """Local symmetry breaking should not reduce the optimal payoff."""
        from rcmas.smt_solve import solve_collective_optimality

        sol_no_sym = solve_collective_optimality(
            territory=territory_L_shape_large,
            num_agents=2,
            horizon=7,
            debug=True,
            symmetry_breaking=False,
        )
        sol_sym = solve_collective_optimality(
            territory=territory_L_shape_large,
            num_agents=2,
            horizon=7,
            debug=True,
            symmetry_breaking=True,
        )
        assert sol_no_sym.is_sat and sol_sym.is_sat
        assert sol_no_sym.payoff_by_agent is not None
        assert sol_sym.payoff_by_agent is not None
        assert sum(sol_sym.payoff_by_agent) == sum(sol_no_sym.payoff_by_agent)


# ---------------------------------------------------------------------------
# SMT integration: disconnected territory symmetry
# ---------------------------------------------------------------------------

class TestDisconnectedSmtIntegration:

    def test_disconnected_with_symmetry(self, territory_disconnected_2_2: Territory):
        from rcmas.smt_solve import solve_collective_optimality

        sol = solve_collective_optimality(
            territory=territory_disconnected_2_2,
            num_agents=2,
            horizon=2,
            debug=True,
            symmetry_breaking=True,
        )
        assert sol.is_sat
        assert sol.payoff_by_agent is not None
        assert sum(sol.payoff_by_agent) == 4  # Each agent gets 2

    def test_disconnected_2x2_with_symmetry(self, territory_disconnected_2x2_2x2: Territory):
        from rcmas.smt_solve import solve_collective_optimality

        sol = solve_collective_optimality(
            territory=territory_disconnected_2x2_2x2,
            num_agents=2,
            horizon=4,
            debug=True,
            symmetry_breaking=True,
        )
        assert sol.is_sat
        assert sol.payoff_by_agent is not None

    def test_disconnected_payoff_preserved(self, territory_disconnected_2_2: Territory):
        from rcmas.smt_solve import solve_collective_optimality

        sol_no_sym = solve_collective_optimality(
            territory=territory_disconnected_2_2,
            num_agents=2,
            horizon=2,
            debug=True,
            symmetry_breaking=False,
        )
        sol_sym = solve_collective_optimality(
            territory=territory_disconnected_2_2,
            num_agents=2,
            horizon=2,
            debug=True,
            symmetry_breaking=True,
        )
        assert sol_no_sym.is_sat and sol_sym.is_sat
        assert sum(sol_sym.payoff_by_agent) == sum(sol_no_sym.payoff_by_agent)

    def test_ibis_disconnected_with_symmetry(self, territory_disconnected_2_2: Territory):
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_disconnected_2_2,
            num_agents=2,
            horizon=2,
            max_iters=10,
            symmetry=True,
        )
        assert res.is_sat


# ---------------------------------------------------------------------------
# CEGAR with Q-IBIS synthesiser
# ---------------------------------------------------------------------------

class TestCegarQibis:

    def test_cegar_qibis_2x2(self, territory_2x2: Territory):
        from rcmas.cegar import solve_cegar

        res = solve_cegar(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            synthesiser="qibis",
        )
        assert res.is_sat
        assert res.found_ne
        assert res.payoff_by_agent is not None

    def test_cegar_qibis_discrete(self, territory_2x2: Territory):
        from rcmas.cegar import solve_cegar

        res = solve_cegar(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            synthesiser="qibis",
            initial_partition="discrete",
        )
        assert res.is_sat
        assert res.found_ne
        assert res.reason == "concrete_fallback"

    def test_cegar_invalid_synthesiser(self, territory_2x2: Territory):
        from rcmas.cegar import solve_cegar

        with pytest.raises(ValueError, match="'ibis', 'qibis', or 'sibis'"):
            solve_cegar(
                territory=territory_2x2,
                num_agents=2,
                horizon=2,
                synthesiser="bogus",
            )

    def test_cegar_ibis_still_works(self, territory_2x2: Territory):
        from rcmas.cegar import solve_cegar

        res = solve_cegar(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            synthesiser="ibis",
        )
        assert res.is_sat
        assert res.found_ne


# ---------------------------------------------------------------------------
# CLI synthesiser flag
# ---------------------------------------------------------------------------

class TestCliSynthesiserFlag:

    def test_parse_cegar_synthesiser_ibis(self):
        from rcmas.cli import _parse_args

        args = _parse_args([
            "cegar", "--grid", "grids/symmetric/2x2.txt",
            "--agents", "2", "--horizon", "2", "--synthesiser", "ibis",
        ])
        assert args.synthesiser == "ibis"

    def test_parse_cegar_synthesiser_qibis(self):
        from rcmas.cli import _parse_args

        args = _parse_args([
            "cegar", "--grid", "grids/symmetric/2x2.txt",
            "--agents", "2", "--horizon", "2", "--synthesiser", "qibis",
        ])
        assert args.synthesiser == "qibis"

    def test_parse_cegar_default_synthesiser(self):
        from rcmas.cli import _parse_args

        args = _parse_args([
            "cegar", "--grid", "grids/symmetric/2x2.txt",
            "--agents", "2", "--horizon", "2",
        ])
        assert args.synthesiser == "ibis"
