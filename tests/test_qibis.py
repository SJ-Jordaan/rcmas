"""Tests for qibis.py (Algorithm 2: Q-IBIS)."""

from __future__ import annotations

import pytest

from rcmas.model import Territory
from rcmas.qibis import QibisConfig, solve_qibis

z3 = pytest.importorskip("z3")


# ===================================================================
# Smoke tests
# ===================================================================

class TestQibisSmokeTests:
    def test_2x2_rl_disabled(self):
        territory = Territory.from_ascii(["..", ".."])
        res = solve_qibis(
            territory=territory, num_agents=2, horizon=2,
            cfg=QibisConfig(max_iters=5, rl_episodes_per_iter=0, timeout_ms=2000),
        )
        assert res.is_sat
        assert res.final_solution is not None
        assert res.final_solution.final_state is not None
        assert all(o != -1 for o in res.final_solution.final_state.owner_by_index)

    def test_2x2_rl_enabled(self):
        territory = Territory.from_ascii(["..", ".."])
        res = solve_qibis(
            territory=territory, num_agents=2, horizon=2,
            cfg=QibisConfig(
                max_iters=5, rl_episodes_per_iter=20, rl_top_k_actions=2,
                timeout_ms=5000, seed=0,
            ),
        )
        assert res.is_sat
        assert res.final_solution is not None
        assert res.final_solution.final_state is not None
        assert all(o != -1 for o in res.final_solution.final_state.owner_by_index)

    def test_1x2_trivial(self):
        """Trivial: 2 agents on 1x2, 1 round."""
        territory = Territory.from_ascii([".."])
        res = solve_qibis(
            territory=territory, num_agents=2, horizon=1,
            cfg=QibisConfig(max_iters=3, rl_episodes_per_iter=0, timeout_ms=2000),
        )
        assert res.is_sat
        assert res.payoff_by_agent is not None
        assert res.payoff_by_agent == (1, 1)


# ===================================================================
# Convergence
# ===================================================================

class TestQibisConvergence:
    def test_terminates_within_max_iters(self):
        territory = Territory.from_ascii(["..", ".."])
        res = solve_qibis(
            territory=territory, num_agents=2, horizon=2,
            cfg=QibisConfig(max_iters=3, rl_episodes_per_iter=0, timeout_ms=2000),
        )
        assert res.iterations <= 3

    def test_reason_is_valid(self):
        territory = Territory.from_ascii(["..", ".."])
        res = solve_qibis(
            territory=territory, num_agents=2, horizon=2,
            cfg=QibisConfig(max_iters=5, rl_episodes_per_iter=0, timeout_ms=2000),
        )
        assert res.reason in ("ne", "cycle", "max_iters", "unsat", "unknown")


# ===================================================================
# Input validation
# ===================================================================

class TestQibisValidation:
    def test_zero_agents_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="num_agents"):
            solve_qibis(territory=territory, num_agents=0, horizon=1, cfg=QibisConfig())

    def test_zero_horizon_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="horizon"):
            solve_qibis(territory=territory, num_agents=1, horizon=0, cfg=QibisConfig())

    def test_zero_max_iters_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="max_iters"):
            solve_qibis(territory=territory, num_agents=1, horizon=1, cfg=QibisConfig(max_iters=0))
