"""Tests for local symmetry (Section 5.2) and disconnected territories (Section 5.3).

The brute-force oracle is in :mod:`tests.test_symmetry`.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest

from rcmas.model import UNOWNED, Territory, rewards_from_owner
from rcmas.symmetry import (
    LocalConstraint,
    component_info,
    find_maximal_rectangular_subregions,
    local_symmetry_info,
    symmetry_breaking_plan,
    territory_connected_components,
)

from .brute_force import final_states
from .test_symmetry import (
    act,
    dominated,
    grid,
    group_closure,
    is_automorphism,
    orbit_report,
)

z3 = pytest.importorskip("z3")

GRID_DIR = Path(__file__).resolve().parent.parent / "grids"


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
    return Territory.from_ascii(
        [
            "..##",
            "....",
            "....",
            "....",
        ]
    )


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

    def test_disconnected_2x2_isomorphic(
        self, territory_disconnected_2x2_2x2: Territory
    ):
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
        # Sectors 0-11 form the 4×3 rectangle (bottom row first, Remark 3.1.1)
        sub = frozenset(range(0, 12))
        lsi = local_symmetry_info(territory_L_shape_large, sub)
        assert len(lsi.automorphisms) == 4  # V_4
        assert len(lsi.orbits) == 4
        assert lsi.boundary == frozenset({8, 9})
        assert len(lsi.boundary_pairs) > 0
        # Representatives should be 4 (one per orbit)
        assert len(lsi.representatives) == 4

    def test_subregion_boundary_computed(self, territory_L_shape_large: Territory):
        sub = frozenset(range(0, 12))
        lsi = local_symmetry_info(territory_L_shape_large, sub)
        # Boundary sectors are those in R adjacent to T\R
        # Sectors 8 and 9 are adjacent to sectors 12 and 13
        assert 8 in lsi.boundary
        assert 9 in lsi.boundary

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

    def test_disconnected_2x2_with_symmetry(
        self, territory_disconnected_2x2_2x2: Territory
    ):
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

    def test_ibis_disconnected_with_symmetry(
        self, territory_disconnected_2_2: Territory
    ):
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

        args = _parse_args(
            [
                "cegar",
                "--grid",
                "grids/symmetric/2x2.txt",
                "--agents",
                "2",
                "--horizon",
                "2",
                "--synthesiser",
                "ibis",
            ]
        )
        assert args.synthesiser == "ibis"

    def test_parse_cegar_synthesiser_qibis(self):
        from rcmas.cli import _parse_args

        args = _parse_args(
            [
                "cegar",
                "--grid",
                "grids/symmetric/2x2.txt",
                "--agents",
                "2",
                "--horizon",
                "2",
                "--synthesiser",
                "qibis",
            ]
        )
        assert args.synthesiser == "qibis"

    def test_parse_cegar_default_synthesiser(self):
        from rcmas.cli import _parse_args

        args = _parse_args(
            [
                "cegar",
                "--grid",
                "grids/symmetric/2x2.txt",
                "--agents",
                "2",
                "--horizon",
                "2",
            ]
        )
        assert args.synthesiser == "ibis"


# ---------------------------------------------------------------------------
# Local symmetry: the examples of Section 5.2 in the tool's numbering
# ---------------------------------------------------------------------------
#
# L-4x4-2 numbered by Remark 3.1.1 (bottom row first):
#
#     12 13 .  .        thesis (top row first):   1  2
#      8  9 10 11                                 3  4  5  6
#      4  5  6  7                                 7  8  9 10
#      0  1  2  3                                11 12 13 14


def _sigma_lr(lsi) -> dict[int, int]:
    """The left-right reflection of R_1 = {0..11}, extended by the identity."""
    (perm,) = [p for p in lsi.automorphisms if p.get(8) == 11 and p.get(0) == 3]
    return {i: perm.get(i, i) for i in range(14)}


def _extend(perm: dict[int, int], S: int) -> list[int]:
    return [perm.get(i, i) for i in range(S)]


class TestLocalExamples:
    def test_subregions_of_l_shape(self, territory_L_shape_large: Territory):
        """The tool's subregions: the 4x3 strip R_1 and the 2x4 strip on the left."""
        subs = find_maximal_rectangular_subregions(territory_L_shape_large)
        assert subs == [frozenset(range(12)), frozenset({0, 1, 4, 5, 8, 9, 12, 13})]

    def test_problematic_pairs(self, territory_L_shape_large: Territory):
        """Examples 5.2.4 and 5.2.6: BP(sigma_LR) = BP(H_R1) = {(3,1), (4,2)},
        and Reps_R1 has one sector in each of the four orbits."""
        lsi = local_symmetry_info(territory_L_shape_large, frozenset(range(12)))
        sigma = _sigma_lr(lsi)
        assert sigma[8] == 11 and sigma[9] == 10
        key = tuple(sigma[i] for i in range(14))
        assert lsi.boundary_pairs[key] == {(8, 12), (9, 13)}
        assert all(bp == {(8, 12), (9, 13)} for bp in lsi.boundary_pairs.values())
        assert len(lsi.boundary_pairs) == 3
        assert lsi.all_boundary_pairs == {(8, 12), (9, 13)}
        assert set(lsi.representatives) == {0, 1, 4, 5}

    def test_reward_drops_without_compatibility(
        self, territory_L_shape_large: Territory
    ):
        """Example 5.2.5: the path {1, 3, 7} spans the pair (3, 1) and loses a sector."""
        t = territory_L_shape_large
        lsi = local_symmetry_info(t, frozenset(range(12)))
        owner = [UNOWNED] * 14
        for i in (12, 8, 4):  # thesis sectors 1, 3, 7
            owner[i] = 0
        lc = LocalConstraint.of(lsi)
        assert any(owner[i] == owner[j] != UNOWNED for i, j in lc.boundary_pairs)
        img, _ = act(_extend(_sigma_lr(lsi), 14), owner, ())
        nb = t.neighbor_map()
        assert rewards_from_owner(tuple(owner), 1, nb) == (3,)
        assert rewards_from_owner(img, 1, nb) == (2,)

    def test_rewards_can_increase(self, territory_L_shape_large: Territory):
        """Remark 5.2.1: a compatible state whose rewards go from (3,3,4) to (4,4,4)."""
        t = territory_L_shape_large
        lsi = local_symmetry_info(t, frozenset(range(12)))
        owner = [UNOWNED] * 14
        for a, region in enumerate(({12, 11, 7, 3}, {13, 10, 6, 2}, {8, 9, 4, 5})):
            for i in region:
                owner[i] = a
        assert not any(
            owner[i] == owner[j] != UNOWNED for i, j in lsi.all_boundary_pairs
        )
        img, _ = act(_extend(_sigma_lr(lsi), 14), owner, ())
        nb = t.neighbor_map()
        assert rewards_from_owner(tuple(owner), 3, nb) == (3, 3, 4)
        assert rewards_from_owner(img, 3, nb) == (4, 4, 4)


class TestRewardMonotonicity:
    @pytest.mark.parametrize("name", ["L-small", "L-4x4-2", "T-5x5"])
    def test_compatible_states_lose_no_reward(self, name: str):
        """Theorem 5.2.2: if no agent controls both sectors of a pair in
        BP(sigma_R), the local extension of sigma_R decreases no reward.
        Checked for every subregion and every element of H_R on random states."""
        t = (
            grid(name)
            if name != "T-5x5"
            else Territory.from_ascii(
                (GRID_DIR / "local_symmetry" / "T-5x5.txt").read_text().split()
            )
        )
        S = len(t)
        nb = t.neighbor_map()
        rng = random.Random(7)
        subs = find_maximal_rectangular_subregions(t)
        assert subs
        checked = 0
        for _ in range(400):
            n = rng.randint(1, 4)
            m = rng.randint(1, S // n)
            labels = [a for a in range(n) for _ in range(m)] + [UNOWNED] * (S - n * m)
            rng.shuffle(labels)
            owner = tuple(labels)
            before = rewards_from_owner(owner, n, nb)
            for sub in subs:
                lsi = local_symmetry_info(t, sub)
                boundary = lsi.boundary
                for perm in lsi.automorphisms:
                    bp = {
                        (i, j)
                        for i in boundary
                        for j in nb[i]
                        if j not in sub and j not in nb[perm[i]]
                    }
                    if any(owner[i] == owner[j] != UNOWNED for i, j in bp):
                        continue
                    img, _ = act(_extend(perm, S), owner, ())
                    after = rewards_from_owner(img, n, nb)
                    assert all(x >= y for x, y in zip(after, before))
                    checked += 1
        assert checked > 1000


# ---------------------------------------------------------------------------
# Corollary 5.2.3 by enumeration
# ---------------------------------------------------------------------------

LOCAL_CASES = [
    ("L-small", 2, [(0, 0), (1, 1), (3, 3), (2, 3), (3, 1)]),
    ("L-small", 3, [(0, 0, 0), (2, 2, 2), (1, 2, 2), (2, 1, 1)]),
    ("L-small", 7, [(0,) * 7, (1,) * 7]),
    ("L-4x4-2", 2, [(0, 0), (5, 5), (7, 7), (7, 5), (4, 7)]),
]


class TestLocalOracle:
    @pytest.mark.parametrize("name,n,demand_list", LOCAL_CASES)
    def test_satisfying_and_optimal_strategies_survive(
        self, name: str, n: int, demand_list
    ):
        """With the constraints L_l (agent-1 consequent, BP(H_R) antecedent),
        a satisfying (collectively optimal) play exists whenever one exists
        (Corollary 5.2.3).  Stronger, as the proof gives: every reward
        profile of a satisfying play, sorted within classes, is dominated by
        that of a constrained one."""
        t = grid(name)
        for d in demand_list:
            plan = symmetry_breaking_plan(t, n, d)
            assert plan.local, "a trivial global group gives local constraints"
            rep = orbit_report(t, n, d, plan)
            assert rep["best_admitted"] == rep["best_all"], (name, n, d)
            for p in rep["profiles_all"]:
                assert any(dominated(p, q) for q in rep["profiles_admitted"]), (
                    name,
                    n,
                    d,
                    p,
                )

    def test_constraint_restricts_agent_one_only(self):
        """With every sector claimed in round 0 (m = 1), the old consequent for
        every agent excludes all plays; the constraint of Corollary 5.2.3 does not."""
        t = grid("L-small")
        plan = symmetry_breaking_plan(t, 7)
        (lc,) = plan.local
        admitted = old = 0
        for owner in final_states(7, 7, 1):
            opens = tuple(owner.index(a) for a in range(7))
            admitted += plan.admits(opens, owner)
            compatible = not any(
                owner[i] == owner[j] != UNOWNED for i, j in lc.boundary_pairs
            )
            old += plan.lex_leader_holds(opens) and not (
                compatible
                and any(
                    o in lc.subregion and o not in lc.representatives for o in opens
                )
            )
        assert admitted > 0 and old == 0

    @pytest.mark.parametrize("n", [2, 8, 14])
    def test_smt_satisfiable_and_optimal_on_l_shape(self, n: int):
        """End to end on L-4x4-2: the old all-agent consequent made n = 8 and
        n = 14 unsatisfiable; the optimum is unchanged."""
        from rcmas.smt_solve import solve_collective_optimality

        t = grid("L-4x4-2")
        m = len(t) // n
        plain = solve_collective_optimality(territory=t, num_agents=n, horizon=m)
        sym = solve_collective_optimality(
            territory=t, num_agents=n, horizon=m, symmetry_breaking=True
        )
        assert plain.is_sat and sym.is_sat
        assert sum(sym.payoff_by_agent) == sum(plain.payoff_by_agent)
        assert symmetry_breaking_plan(t, n).admits(
            sym.play.actions[0], sym.play.final_state
        )


# ---------------------------------------------------------------------------
# Disconnected territories: classes of copies and the group H (Theorem 5.3.2)
# ---------------------------------------------------------------------------


def _disconnected_grid_files() -> list[Path]:
    files = sorted((GRID_DIR / "disconnected").glob("*.txt"))
    files += sorted((GRID_DIR / "random").glob("rand_d_*.txt"))
    return files


def _check_hypotheses(t: Territory) -> None:
    """Hypotheses (i) and (ii) of Theorem 5.3.2 for the tool's labels, and
    that each order-preserving bijection between copies is an isomorphism."""
    ci = component_info(t)
    nb = t.neighbor_map()
    for cls in ci.copy_classes:
        copies = [sorted(ci.components[c]) for c in cls]
        for k, first in enumerate(copies):
            assert [ci.labels[y] for y in first] == list(range(len(first)))  # (i)
            for second in copies[k + 1 :]:
                for x in range(len(first)):
                    for z in range(x, len(first)):
                        assert first[x] < second[z]  # (ii)
                pos = {s: r for r, s in enumerate(second)}
                for r, y in enumerate(first):
                    assert {pos[j] for j in nb[second[r]]} == {
                        copies[k].index(j) for j in nb[y]
                    }


class TestComponentClasses:
    @pytest.mark.parametrize("path", _disconnected_grid_files(), ids=lambda p: p.stem)
    def test_benchmark_numbering_meets_hypotheses(self, path: Path):
        t = Territory.from_ascii(path.read_text().split())
        ci = component_info(t)
        assert len(ci.components) > 1
        _check_hypotheses(t)
        # Side-by-side copies are translates: each shape group is one class.
        assert len(ci.copy_classes) == len(ci.iso_classes)

    def test_rotated_copy_forms_its_own_class(self):
        """A 3x2 rectangle and its rotation are congruent, but their
        order-preserving bijection is no isomorphism, so (i) cannot hold for
        one class: the tool keeps them in separate classes."""
        t = grid("rot")
        ci = component_info(t)
        assert ci.iso_classes == ((0, 1),)
        assert ci.copy_classes == ((0,), (1,))
        assert ci.consecutive_copies() == ()
        _check_hypotheses(t)

    @pytest.mark.parametrize(
        "name", ["2x2-2x2", "3x2-2x1", "3comp-2x1", "rot", "stack", "7x4-gap"]
    )
    def test_group_h_consists_of_automorphisms(self, name: str):
        t = grid(name)
        ci = component_info(t)
        group = group_closure(ci.generators, len(t))
        assert all(is_automorphism(t, h) for h in group)
        orbits = {frozenset(h[i] for h in group) for i in range(len(t))}
        assert orbits == set(ci.orbits)

    def test_example_7x4(self):
        """Examples 5.3.1-5.3.2 and 5.4.1: |H| = |V_4 wr S_2| = 32, four
        orbits, and the representatives lie in the first copy."""
        t = grid("7x4-gap")
        ci = component_info(t)
        assert len(group_closure(ci.generators, len(t))) == 32
        assert ci.copy_classes == ((0, 1),)
        assert ci.representatives == (0, 1, 6, 7)
        plan = symmetry_breaking_plan(t, 2)
        assert plan.copy_pairs == ci.consecutive_copies()
        owner = (UNOWNED,) * len(t)
        # Agent 1 at label 1 of K_1, agent 2 at label 0 of K_2: (3) fails ...
        assert plan.lex_leader_holds((1, 3)) and 1 in plan.representatives
        assert not plan.admits((1, 3), owner)
        # ... and exchanging the copies and the agents repairs it.
        assert plan.admits((0, 4), owner)

    def test_inter_copy_constraint_can_exclude_an_allocation(self):
        """Remark 5.1.5: on the row with copies 3-6 of a one-sector
        component, the allocation {1,2},{3,4},{5,6} meets no constraint set,
        while its image {1,2},{3,5},{4,6} under the exchange of 4 and 5 does."""
        t = grid("row-6")
        plan = symmetry_breaking_plan(t, 3)

        def admitted_up_to_labels(regions):
            for perm in itertools.permutations(regions):
                owner = [UNOWNED] * 6
                for a, reg in enumerate(perm):
                    for i in reg:
                        owner[i] = a
                if any(plan.admits(o, tuple(owner)) for o in itertools.product(*perm)):
                    return True
            return False

        assert not admitted_up_to_labels(([0, 1], [2, 3], [4, 5]))
        assert admitted_up_to_labels(([0, 1], [2, 4], [3, 5]))


DISCONNECTED_CASES = [
    ("2x2-2x2", 2, [(0, 0), (4, 4), (2, 4)]),
    ("2x2-2x2", 4, [(0,) * 4, (2, 2, 2, 2), (1, 1, 2, 2), (2, 1, 1, 1)]),
    ("2x2-2x2", 6, [(0,) * 6, (1,) * 6]),
    ("2x2-2x2", 8, [(0,) * 8]),
    ("3x2-2x1", 2, [(0, 0), (4, 4), (5, 2)]),
    ("3comp-2x1", 2, [(0, 0), (2, 2), (1, 2)]),
    ("3comp-2x1", 3, [(0, 0, 0), (2, 2, 2), (1, 2, 2)]),
    ("rot", 2, [(0, 0), (6, 6)]),
    ("stack", 2, [(0, 0), (6, 6), (3, 5)]),
    ("row-6", 3, [(0, 0, 0), (1, 1, 2)]),
]


class TestDisconnectedOracle:
    @pytest.mark.parametrize("name,n,demand_list", DISCONNECTED_CASES)
    def test_every_orbit_keeps_a_member(self, name: str, n: int, demand_list):
        """Theorem 5.3.2: every G_H-orbit of satisfying plays contains a play
        that meets (1) the lex-leader constraint, (2) spatial
        canonicalisation under H and (3) the min-based ordering of the
        first-round actions of agent 1's class in consecutive copies."""
        t = grid(name)
        for d in demand_list:
            plan = symmetry_breaking_plan(t, n, d)
            rep = orbit_report(t, n, d, plan)
            assert rep["orbit_failures"] == 0, (name, n, d)
            assert rep["best_admitted"] == rep["best_all"], (name, n, d)

    def test_old_rule_lost_every_play(self):
        """The examined rule (no larger rank in an earlier copy than in a
        later one, over all agents) admits no play on 2x2-2x2 with n = 6."""
        t = grid("2x2-2x2")
        ci = component_info(t)
        (cls,) = ci.copy_classes
        first, second = (sorted(ci.components[c]) for c in cls)
        rank = {s: r for c in (first, second) for r, s in enumerate(c)}
        plan = symmetry_breaking_plan(t, 6)
        new = old = 0
        for owner in final_states(8, 6, 1):
            opens = tuple(owner.index(a) for a in range(6))
            new += plan.admits(opens, owner)
            in_first = [rank[o] for o in opens if o in first]
            in_second = [rank[o] for o in opens if o in second]
            old += plan.lex_leader_holds(opens) and not (
                in_first and in_second and max(in_first) > min(in_second)
            )
        assert new > 0 and old == 0

    @pytest.mark.parametrize(
        "name,n,demands",
        [
            ("2x2-2x2", 2, None),
            ("2x2-2x2", 4, (2, 2, 2, 2)),
            ("2x2-2x2", 6, None),
            ("2x2-2x2", 7, None),
            ("2x2-2x2", 8, None),
            ("3x2-2x1", 3, None),
            ("3x2-2x1", 5, (2, 2, 2, 2, 2)),
        ],
    )
    def test_smt_satisfiable_and_optimal(self, name: str, n: int, demands):
        """End to end: smt(F and [Sym], f) has the optimum of smt(F, f), and
        the returned play meets the constraints."""
        from rcmas.smt_solve import solve_collective_optimality

        t = grid(name)
        m = len(t) // n
        plain = solve_collective_optimality(
            territory=t, num_agents=n, horizon=m, demands=demands
        )
        sym = solve_collective_optimality(
            territory=t,
            num_agents=n,
            horizon=m,
            demands=demands,
            symmetry_breaking=True,
        )
        assert plain.is_sat and sym.is_sat
        assert sum(sym.payoff_by_agent) == sum(plain.payoff_by_agent)
        plan = symmetry_breaking_plan(t, n, demands)
        assert plan.admits(sym.play.actions[0], sym.play.final_state)

    def test_benchmark_3x3_3x3_six_agents_satisfiable(self):
        """3x3-3x3 with n = 6, where the examined rule blocked deviations."""
        from rcmas.smt_solve import solve_satisfying

        t = Territory.from_ascii(
            (GRID_DIR / "disconnected" / "3x3-3x3.txt").read_text().split()
        )
        sol = solve_satisfying(
            territory=t, num_agents=6, horizon=3, symmetry_breaking=True
        )
        assert sol.is_sat
        assert symmetry_breaking_plan(t, 6).admits(
            sol.play.actions[0], sol.play.final_state
        )
