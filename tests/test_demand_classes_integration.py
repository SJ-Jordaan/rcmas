"""Integration tests for demand-class symmetry breaking across all synthesis paths.

Covers:
  - CO solver with heterogeneous demands + symmetry (payoff preservation)
  - IBIS with heterogeneous demands (3x3, 3 agents)
  - CEGAR with demands threaded through abstract + concrete + verification
  - CLI end-to-end via main() with --demands
  - Correctness oracle: within-class lex-leader does not prune valid NE
"""

from __future__ import annotations

import pytest

from rcmas.model import Territory

from .brute_force import is_equilibrium

z3 = pytest.importorskip("z3")


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


# ---------------------------------------------------------------------------
# CO solver: payoff preservation with heterogeneous demands
# ---------------------------------------------------------------------------


class TestCOWithDemandClasses:
    """Symmetry breaking with demand classes must not reduce optimal payoff."""

    def test_payoff_preserved_heterogeneous_2x2(self, territory_2x2: Territory):
        """demands=(1,2) on 2x2: sym payoff == no-sym payoff (Corollary 5.1.4)."""
        from rcmas.smt_solve import solve_collective_optimality

        no_sym = solve_collective_optimality(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            debug=True,
            symmetry_breaking=False,
            demands=(1, 2),
        )
        with_sym = solve_collective_optimality(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            debug=True,
            symmetry_breaking=True,
            demands=(1, 2),
        )
        assert no_sym.is_sat and with_sym.is_sat
        assert no_sym.payoff_by_agent is not None
        assert with_sym.payoff_by_agent is not None
        assert sum(with_sym.payoff_by_agent) == sum(no_sym.payoff_by_agent)

    def test_payoff_preserved_heterogeneous_3x3(self, territory_3x3: Territory):
        """demands=(2,2,3) on 3x3: sym payoff == no-sym payoff.

        Demands are part of F ([phi_sat]), so both queries use the same
        demands, and each demand is at most m = 3."""
        from rcmas.smt_solve import solve_collective_optimality

        no_sym = solve_collective_optimality(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            debug=True,
            symmetry_breaking=False,
            demands=(2, 2, 3),
        )
        with_sym = solve_collective_optimality(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            debug=True,
            symmetry_breaking=True,
            demands=(2, 2, 3),
        )
        assert no_sym.is_sat and with_sym.is_sat
        assert no_sym.payoff_by_agent is not None
        assert with_sym.payoff_by_agent is not None
        assert sum(with_sym.payoff_by_agent) == sum(no_sym.payoff_by_agent)

    def test_payoff_preserved_all_different_3x3(self, territory_3x3: Territory):
        """All-different demands (1,2,3): only spatial canon, no agent ordering.
        Must still match the unbroken optimal."""
        from rcmas.smt_solve import solve_collective_optimality

        no_sym = solve_collective_optimality(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            debug=True,
            symmetry_breaking=False,
            demands=(1, 2, 3),
        )
        with_sym = solve_collective_optimality(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            debug=True,
            symmetry_breaking=True,
            demands=(1, 2, 3),
        )
        assert no_sym.is_sat and with_sym.is_sat
        assert no_sym.payoff_by_agent is not None
        assert with_sym.payoff_by_agent is not None
        assert sum(with_sym.payoff_by_agent) == sum(no_sym.payoff_by_agent)

    def test_uniform_demands_matches_no_demands(self, territory_2x2: Territory):
        """demands=(0,0) is the game of demands=None, with the same single
        demand class, so the optima agree."""
        from rcmas.smt_solve import solve_collective_optimality

        sol_none = solve_collective_optimality(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            debug=True,
            symmetry_breaking=True,
            demands=None,
        )
        sol_uniform = solve_collective_optimality(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            debug=True,
            symmetry_breaking=True,
            demands=(0, 0),
        )
        assert sol_none.is_sat and sol_uniform.is_sat
        assert sol_none.payoff_by_agent is not None
        assert sol_uniform.payoff_by_agent is not None
        assert sum(sol_none.payoff_by_agent) == sum(sol_uniform.payoff_by_agent)


# ---------------------------------------------------------------------------
# IBIS: heterogeneous demands on 3x3 and 2x3
# ---------------------------------------------------------------------------


class TestIbisWithDemandClasses:
    """IBIS with demand-class lex-leader finds NE correctly."""

    def test_3x3_three_agents_heterogeneous(self, territory_3x3: Territory):
        """3x3, 3 agents, demands=(2,2,3): IBIS with symmetry returns a
        satisfying play, and an equilibrium only if it is one."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            max_iters=10,
            symmetry=True,
            demands=(2, 2, 3),
        )
        assert res.is_sat
        assert res.payoff_by_agent is not None
        assert len(res.payoff_by_agent) == 3
        assert all(r >= d for r, d in zip(res.payoff_by_agent, (2, 2, 3)))
        if res.found_ne:
            assert is_equilibrium(res.play, territory_3x3.neighbor_map(), (2, 2, 3))

    def test_demands_above_horizon_unsatisfiable(self, territory_3x3: Territory):
        """demands=(2,2,5) with m = 3: no satisfying play, with or without symmetry."""
        from rcmas.ibis import solve_ibis

        for flag in (False, True):
            res = solve_ibis(
                territory=territory_3x3,
                num_agents=3,
                horizon=3,
                max_iters=10,
                symmetry=flag,
                demands=(2, 2, 5),
            )
            assert not res.is_sat and res.outcome == "uns"

    def test_3x3_three_agents_all_different(self, territory_3x3: Territory):
        """3x3, 3 agents, demands=(1,2,3): all-different means no agent lex-leader."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            max_iters=10,
            symmetry=True,
            demands=(1, 2, 3),
        )
        assert res.is_sat
        assert res.payoff_by_agent is not None

    def test_2x3_two_agents_heterogeneous(self, territory_2x3: Territory):
        """2x3 (6 sectors), 2 agents, demands=(2,3): asymmetric demand classes."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_2x3,
            num_agents=2,
            horizon=3,
            max_iters=10,
            symmetry=True,
            demands=(2, 3),
        )
        assert res.is_sat
        assert res.payoff_by_agent is not None

    def test_ibis_ne_sound_with_and_without_symmetry(self, territory_3x3: Territory):
        """demands=(3,3,3): with and without symmetry, every reported
        equilibrium is one (checked by enumeration).  The two runs may return
        different equilibria, since symmetry changes only the initial play
        (Remark 5.1.4, Section 5.4.5), so their rewards are not compared."""
        from rcmas.ibis import solve_ibis

        d = (3, 3, 3)
        for flag in (False, True):
            res = solve_ibis(
                territory=territory_3x3,
                num_agents=3,
                horizon=3,
                max_iters=15,
                symmetry=flag,
                demands=d,
            )
            assert res.is_sat
            assert res.payoff_by_agent is not None
            assert all(r >= 3 for r in res.payoff_by_agent)
            if res.found_ne:
                assert is_equilibrium(res.play, territory_3x3.neighbor_map(), d)

    def test_2x3_three_agents_two_classes(self, territory_2x3: Territory):
        """2x3, 3 agents, demands=(2,2,2): two agents share a class."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_2x3,
            num_agents=3,
            horizon=2,
            max_iters=10,
            symmetry=True,
            demands=(2, 2, 2),
        )
        assert res.is_sat
        assert res.payoff_by_agent is not None
        assert len(res.payoff_by_agent) == 3


# ---------------------------------------------------------------------------
# CEGAR: demands threaded through abstract/concrete/verify
# ---------------------------------------------------------------------------


class TestCegarWithDemandClasses:
    """CEGAR loop with demand-class lex-leader constraints."""

    def test_cegar_orbit_with_demands_2x2(self, territory_2x2: Territory):
        """CEGAR orbit partition with demands=(1,1) on 2x2."""
        from rcmas.cegar import solve_cegar

        result = solve_cegar(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            initial_partition="orbit",
            symmetry=True,
            demands=(1, 1),
        )
        assert result.is_sat
        assert result.found_ne
        assert result.payoff_by_agent is not None

    def test_cegar_orbit_heterogeneous_3x3(self, territory_3x3: Territory):
        """CEGAR orbit partition with heterogeneous demands=(2,2,3) on 3x3."""
        from rcmas.cegar import solve_cegar

        result = solve_cegar(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            initial_partition="orbit",
            symmetry=True,
            demands=(2, 2, 3),
        )
        assert result.is_sat
        assert result.found_ne
        assert result.payoff_by_agent is not None
        assert len(result.payoff_by_agent) == 3

    def test_cegar_discrete_with_demands(self, territory_2x2: Territory):
        """At the discrete partition the inner S-IBIS runs on M itself, and its
        play is verified by VerifyNE before it is returned (Algorithm 4,
        lines 18-20).  The old tool returned it unchecked as a
        ``concrete_fallback``."""
        from rcmas.cegar import solve_cegar

        result = solve_cegar(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            initial_partition="discrete",
            symmetry=True,
            demands=(2, 2),
        )
        assert result.is_sat
        assert result.found_ne
        assert result.outcome == "ne" and result.reason == "ne"
        assert result.at_discrete

    def test_cegar_demands_none_backward_compatible(self, territory_2x2: Territory):
        """CEGAR with demands=None should work identically to before."""
        from rcmas.cegar import solve_cegar

        result = solve_cegar(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            symmetry=True,
            demands=None,
        )
        assert result.is_sat
        assert result.found_ne


# ---------------------------------------------------------------------------
# Correctness oracle: lex-leader within classes vs global
# ---------------------------------------------------------------------------


class TestLexLeaderCorrectness:
    """Verify that within-class lex-leader does not prune valid solutions
    that the global lex-leader would also find, and does not reject solutions
    valid under within-class ordering."""

    def test_within_class_admits_interleaved_actions(self, territory_2x3: Territory):
        """With demands=(2,3), action[0][0] is unconstrained relative
        to action[1][0], since they are in different classes. This should
        still find a valid CO solution."""
        from rcmas.smt_solve import solve_smt_game

        sol = solve_smt_game(
            territory=territory_2x3,
            num_agents=2,
            horizon=3,
            objective="sum",
            debug=True,
            symmetry_breaking=True,
            demands=(2, 3),
        )
        assert sol.is_sat
        assert sol.payoff_by_agent is not None
        # Total sectors = 6, both agents claim all -> payoffs should sum to 6
        assert sum(sol.payoff_by_agent) == 6

    def test_three_classes_no_over_constraint(self, territory_3x3: Territory):
        """demands=(1,2,3): 3 singleton classes -> 0 lex-leader inequalities.
        Solution space is same as no agent symmetry breaking."""
        from rcmas.smt_solve import solve_smt_game

        sol_full_sym = solve_smt_game(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            objective="sum",
            debug=True,
            symmetry_breaking=True,
            demands=(1, 2, 3),
        )
        sol_no_sym = solve_smt_game(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            objective="sum",
            debug=True,
            symmetry_breaking=False,
            demands=(1, 2, 3),
        )
        assert sol_full_sym.is_sat and sol_no_sym.is_sat
        assert sol_full_sym.payoff_by_agent is not None
        assert sol_no_sym.payoff_by_agent is not None
        # All-different demands means no agent ordering -> same optimal
        assert sum(sol_full_sym.payoff_by_agent) == sum(sol_no_sym.payoff_by_agent)

    def test_two_classes_fewer_constraints(self, territory_2x3: Territory):
        """demands=None (uniform): 2 lex-leader inequalities (global S_3).
        demands=(1,2,2): 1 inequality (one class of 2).  In both games the
        constraints keep the collective optimum (Corollary 5.1.4)."""
        from rcmas.smt_solve import solve_collective_optimality
        from rcmas.smt_solve import symmetry_constraints
        from rcmas.smt_variables import create_variables

        for demands, inequalities in ((None, 2), ((1, 2, 2), 1)):
            sols = [
                solve_collective_optimality(
                    territory=territory_2x3,
                    num_agents=3,
                    horizon=2,
                    debug=True,
                    symmetry_breaking=flag,
                    demands=demands,
                )
                for flag in (False, True)
            ]
            assert all(sol.is_sat for sol in sols)
            assert sum(sols[0].payoff_by_agent) == sum(sols[1].payoff_by_agent)

            posted: list = []

            class _Collect:
                def add(self, *exprs):
                    posted.extend(exprs)

            v = create_variables(territory_2x3, 3, 2)
            symmetry_constraints(_Collect(), v, territory_2x3, demands)
            lex = [e for e in posted if z3.is_le(e)]
            assert len(lex) == inequalities


# ---------------------------------------------------------------------------
# CLI end-to-end with main()
# ---------------------------------------------------------------------------


class TestCliEndToEnd:
    """Exercise main() with --demands for each subcommand."""

    def test_cli_co_with_demands(self, tmp_path):
        """rcmas co --demands works end-to-end."""
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main(
            [
                "co",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--horizon",
                "2",
                "--symmetry",
                "--demands",
                "2,2",
            ]
        )
        assert rc == 0

    def test_cli_ibis_with_demands(self, tmp_path):
        """rcmas ibis --demands works end-to-end."""
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main(
            [
                "ibis",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--horizon",
                "2",
                "--symmetry",
                "--demands",
                "1,2",
            ]
        )
        assert rc == 0

    def test_cli_cegar_with_demands(self, tmp_path):
        """rcmas cegar --demands works end-to-end."""
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main(
            [
                "cegar",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--horizon",
                "2",
                "--symmetry",
                "--demands",
                "2,2",
            ]
        )
        assert rc == 0

    def test_cli_ibis_without_demands(self, tmp_path):
        """Backward compatible: ibis without --demands still works."""
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main(
            [
                "ibis",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--horizon",
                "2",
                "--symmetry",
            ]
        )
        assert rc == 0
