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
from rcmas.symmetry import demand_classes

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
        """demands=(1,2) on 2x2: sym payoff == no-sym payoff."""
        from rcmas.smt_solve import solve_collective_optimality

        no_sym = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=False,
        )
        with_sym = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True, demands=(1, 2),
        )
        assert no_sym.is_sat and with_sym.is_sat
        assert no_sym.payoff_by_agent is not None
        assert with_sym.payoff_by_agent is not None
        assert sum(with_sym.payoff_by_agent) == sum(no_sym.payoff_by_agent)

    def test_payoff_preserved_heterogeneous_3x3(self, territory_3x3: Territory):
        """demands=(2,2,5) on 3x3: sym payoff == no-sym payoff."""
        from rcmas.smt_solve import solve_collective_optimality

        no_sym = solve_collective_optimality(
            territory=territory_3x3, num_agents=3, horizon=3,
            debug=True, symmetry_breaking=False,
        )
        with_sym = solve_collective_optimality(
            territory=territory_3x3, num_agents=3, horizon=3,
            debug=True, symmetry_breaking=True, demands=(2, 2, 5),
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
            territory=territory_3x3, num_agents=3, horizon=3,
            debug=True, symmetry_breaking=False,
        )
        with_sym = solve_collective_optimality(
            territory=territory_3x3, num_agents=3, horizon=3,
            debug=True, symmetry_breaking=True, demands=(1, 2, 3),
        )
        assert no_sym.is_sat and with_sym.is_sat
        assert no_sym.payoff_by_agent is not None
        assert with_sym.payoff_by_agent is not None
        assert sum(with_sym.payoff_by_agent) == sum(no_sym.payoff_by_agent)

    def test_uniform_demands_matches_no_demands(self, territory_2x2: Territory):
        """Uniform demands=(2,2) should give the same optimal payoff as demands=None."""
        from rcmas.smt_solve import solve_collective_optimality

        sol_none = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True, demands=None,
        )
        sol_uniform = solve_collective_optimality(
            territory=territory_2x2, num_agents=2, horizon=2,
            debug=True, symmetry_breaking=True, demands=(2, 2),
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
        """3x3, 3 agents, demands=(2,2,5): IBIS finds NE with symmetry."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_3x3, num_agents=3, horizon=3,
            max_iters=10, symmetry=True, demands=(2, 2, 5),
        )
        assert res.is_sat
        assert res.payoff_by_agent is not None
        assert len(res.payoff_by_agent) == 3

    def test_3x3_three_agents_all_different(self, territory_3x3: Territory):
        """3x3, 3 agents, demands=(1,2,3): all-different means no agent lex-leader."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_3x3, num_agents=3, horizon=3,
            max_iters=10, symmetry=True, demands=(1, 2, 3),
        )
        assert res.is_sat
        assert res.payoff_by_agent is not None

    def test_2x3_two_agents_heterogeneous(self, territory_2x3: Territory):
        """2x3 (6 sectors), 2 agents, demands=(2,4): asymmetric demand classes."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_2x3, num_agents=2, horizon=3,
            max_iters=10, symmetry=True, demands=(2, 4),
        )
        assert res.is_sat
        assert res.payoff_by_agent is not None

    def test_ibis_ne_preserved_with_demands(self, territory_3x3: Territory):
        """NE found with demands+sym has same total payoff as without symmetry."""
        from rcmas.ibis import solve_ibis

        res_no_sym = solve_ibis(
            territory=territory_3x3, num_agents=3, horizon=3,
            max_iters=15, symmetry=False,
        )
        res_sym = solve_ibis(
            territory=territory_3x3, num_agents=3, horizon=3,
            max_iters=15, symmetry=True, demands=(3, 3, 3),
        )
        assert res_no_sym.is_sat and res_sym.is_sat
        assert res_no_sym.payoff_by_agent is not None
        assert res_sym.payoff_by_agent is not None
        # NE payoffs should match (both find an NE on same game)
        assert sum(res_sym.payoff_by_agent) == sum(res_no_sym.payoff_by_agent)

    def test_2x3_three_agents_two_classes(self, territory_2x3: Territory):
        """2x3, 3 agents, demands=(2,2,2): two agents share a class."""
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory_2x3, num_agents=3, horizon=2,
            max_iters=10, symmetry=True, demands=(2, 2, 2),
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
            territory=territory_2x2, num_agents=2, horizon=2,
            initial_partition="orbit", symmetry=True, demands=(1, 1),
        )
        assert result.is_sat
        assert result.found_ne
        assert result.payoff_by_agent is not None

    def test_cegar_orbit_heterogeneous_3x3(self, territory_3x3: Territory):
        """CEGAR orbit partition with heterogeneous demands=(2,2,5) on 3x3."""
        from rcmas.cegar import solve_cegar

        result = solve_cegar(
            territory=territory_3x3, num_agents=3, horizon=3,
            initial_partition="orbit", symmetry=True, demands=(2, 2, 5),
        )
        assert result.is_sat
        assert result.found_ne
        assert result.payoff_by_agent is not None
        assert len(result.payoff_by_agent) == 3

    def test_cegar_discrete_with_demands(self, territory_2x2: Territory):
        """Discrete partition with demands falls through to concrete IBIS."""
        from rcmas.cegar import solve_cegar

        result = solve_cegar(
            territory=territory_2x2, num_agents=2, horizon=2,
            initial_partition="discrete", symmetry=True, demands=(2, 2),
        )
        assert result.is_sat
        assert result.found_ne
        assert result.reason == "concrete_fallback"

    def test_cegar_demands_none_backward_compatible(self, territory_2x2: Territory):
        """CEGAR with demands=None should work identically to before."""
        from rcmas.cegar import solve_cegar

        result = solve_cegar(
            territory=territory_2x2, num_agents=2, horizon=2,
            symmetry=True, demands=None,
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
        """With demands=(2,4), only action[0][0] is unconstrained relative
        to action[1][0], since they are in different classes. This should
        still find a valid CO solution."""
        from rcmas.smt_solve import solve_smt_game

        sol = solve_smt_game(
            territory=territory_2x3, num_agents=2, horizon=3,
            objective="sum", debug=True, symmetry_breaking=True,
            demands=(2, 4),
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
            territory=territory_3x3, num_agents=3, horizon=3,
            objective="sum", debug=True, symmetry_breaking=True,
            demands=(1, 2, 3),
        )
        sol_no_sym = solve_smt_game(
            territory=territory_3x3, num_agents=3, horizon=3,
            objective="sum", debug=True, symmetry_breaking=False,
        )
        assert sol_full_sym.is_sat and sol_no_sym.is_sat
        assert sol_full_sym.payoff_by_agent is not None
        assert sol_no_sym.payoff_by_agent is not None
        # All-different demands means no agent ordering -> same optimal
        assert sum(sol_full_sym.payoff_by_agent) == sum(sol_no_sym.payoff_by_agent)

    def test_two_classes_fewer_constraints(self, territory_2x3: Territory):
        """demands=None (uniform): 2 lex-leader inequalities (global S_3).
        demands=(1,2,2): 1 inequality (one class of 2).
        The fewer-constraints version should find at least as good a payoff."""
        from rcmas.smt_solve import solve_collective_optimality

        sol_global = solve_collective_optimality(
            territory=territory_2x3, num_agents=3, horizon=2,
            debug=True, symmetry_breaking=True, demands=None,
        )
        sol_classes = solve_collective_optimality(
            territory=territory_2x3, num_agents=3, horizon=2,
            debug=True, symmetry_breaking=True, demands=(1, 2, 2),
        )
        assert sol_global.is_sat and sol_classes.is_sat
        assert sol_global.payoff_by_agent is not None
        assert sol_classes.payoff_by_agent is not None
        # Fewer constraints means at least as large a feasible set
        assert sum(sol_classes.payoff_by_agent) >= sum(sol_global.payoff_by_agent)


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
        rc = main([
            "co", "--grid", str(grid), "--agents", "2", "--horizon", "2",
            "--symmetry", "--demands", "2,2",
        ])
        assert rc == 0

    def test_cli_ibis_with_demands(self, tmp_path):
        """rcmas ibis --demands works end-to-end."""
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main([
            "ibis", "--grid", str(grid), "--agents", "2", "--horizon", "2",
            "--symmetry", "--demands", "1,2",
        ])
        assert rc == 0

    def test_cli_cegar_with_demands(self, tmp_path):
        """rcmas cegar --demands works end-to-end."""
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main([
            "cegar", "--grid", str(grid), "--agents", "2", "--horizon", "2",
            "--symmetry", "--demands", "2,2",
        ])
        assert rc == 0

    def test_cli_ibis_without_demands(self, tmp_path):
        """Backward compatible: ibis without --demands still works."""
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main([
            "ibis", "--grid", str(grid), "--agents", "2", "--horizon", "2",
            "--symmetry",
        ])
        assert rc == 0
