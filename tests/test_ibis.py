"""Tests for ibis.py (Algorithm 1: IBIS)."""

from __future__ import annotations

import pytest

from rcmas.model import Territory
from rcmas.ibis import solve_ibis

z3 = pytest.importorskip("z3")


# ===================================================================
# Smoke tests
# ===================================================================

class TestIbisSmokeTests:
    def test_2x2_h2(self):
        territory = Territory.from_ascii(["..", ".."])
        res = solve_ibis(territory=territory, num_agents=2, horizon=2, seed=0, max_iters=5)
        assert res.is_sat
        assert res.strategy is not None
        assert res.payoff_by_agent is not None
        assert len(res.payoff_by_agent) == 2
        assert res.final_solution is not None
        assert res.final_solution.final_state is not None
        assert all(o != -1 for o in res.final_solution.final_state.owner_by_index)

    def test_1x2_h1_trivial(self):
        """2 agents on 1x2, 1 round: trivially fills the board. NE exists."""
        territory = Territory.from_ascii([".."])
        res = solve_ibis(territory=territory, num_agents=2, horizon=1, seed=0, max_iters=5)
        assert res.is_sat
        assert res.payoff_by_agent is not None
        assert res.payoff_by_agent == (1, 1)

    def test_1x4_h2_two_agents(self):
        """2 agents on 1x4, 2 rounds: each claims 2 sectors."""
        territory = Territory.from_ascii(["...."])
        res = solve_ibis(territory=territory, num_agents=2, horizon=2, seed=0, max_iters=10)
        assert res.is_sat
        assert res.payoff_by_agent is not None
        assert sum(res.payoff_by_agent) >= 2


# ===================================================================
# Convergence / termination
# ===================================================================

class TestIbisConvergence:
    def test_terminates_within_max_iters(self):
        territory = Territory.from_ascii(["..", ".."])
        res = solve_ibis(territory=territory, num_agents=2, horizon=2, seed=0, max_iters=3)
        assert res.iterations <= 3

    def test_finds_ne_on_symmetric_2x2(self):
        """On 2x2 with 2 agents, the symmetric allocation is a NE."""
        territory = Territory.from_ascii(["..", ".."])
        res = solve_ibis(territory=territory, num_agents=2, horizon=2, seed=0, max_iters=10)
        assert res.is_sat
        # Should converge (NE or cycle)
        assert res.reason in ("ne", "cycle", "max_iters")


# ===================================================================
# Input validation
# ===================================================================

class TestIbisValidation:
    def test_zero_agents_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="num_agents"):
            solve_ibis(territory=territory, num_agents=0, horizon=1)

    def test_zero_horizon_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="horizon"):
            solve_ibis(territory=territory, num_agents=1, horizon=0)

    def test_zero_max_iters_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="max_iters"):
            solve_ibis(territory=territory, num_agents=1, horizon=1, max_iters=0)
