"""Tests for smt_objectives.py (Def 24-25)."""

from __future__ import annotations

import pytest

from rcmas.model import Territory
from rcmas.smt_variables import create_variables
from rcmas.smt_constraints import (
    adjacency_constraint,
    cohesive_region_constraint,
    collision_constraint,
    evolution_constraint,
    init_constraint,
    protocol_constraint,
    reward_constraint,
    size_constraint,
)
from rcmas.smt_objectives import qualitative_objective, quantitative_objective

z3 = pytest.importorskip("z3")


def _build_full_model(territory, num_agents, horizon):
    """Build solver with all core constraints (no victory)."""
    opt = z3.Optimize()
    v = create_variables(territory, num_agents, horizon)
    init_constraint(opt, v)
    protocol_constraint(opt, v)
    collision_constraint(opt, v)
    evolution_constraint(opt, v)
    adjacency_constraint(opt, v)
    cohesive_region_constraint(opt, v)
    size_constraint(opt, v)
    reward_constraint(opt, v)
    return opt, v


def _model_val(m, expr) -> int:
    return m.eval(expr, model_completion=True).as_long()


# ===================================================================
# Def 24: Qualitative objective (maximise sum of payoffs)
# ===================================================================

class TestQualitativeObjective:
    def test_maximises_sum_2x2(self):
        territory = Territory.from_ascii(["..", ".."])
        opt, v = _build_full_model(territory, num_agents=2, horizon=2)
        qualitative_objective(opt, v)
        assert opt.check() == z3.sat
        m = opt.model()
        p0 = _model_val(m, v.payoff[0])
        p1 = _model_val(m, v.payoff[1])
        assert p0 + p1 == 4

    def test_single_agent_gets_full_territory(self):
        """1 agent, 1x3, 3 rounds: maximised sum = 3."""
        territory = Territory.from_ascii(["..."])
        opt, v = _build_full_model(territory, num_agents=1, horizon=3)
        qualitative_objective(opt, v)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.payoff[0]) == 3

    def test_two_agents_linear_territory(self):
        """2 agents on 1x4, 2 rounds: 4 sectors, each agent claims 2.
        Optimal: each gets a connected pair -> sum = 4."""
        territory = Territory.from_ascii(["...."])
        opt, v = _build_full_model(territory, num_agents=2, horizon=2)
        qualitative_objective(opt, v)
        assert opt.check() == z3.sat
        m = opt.model()
        p0 = _model_val(m, v.payoff[0])
        p1 = _model_val(m, v.payoff[1])
        assert p0 + p1 == 4
        assert p0 >= 1
        assert p1 >= 1


# ===================================================================
# Def 25: Quantitative objective (maximise single agent)
# ===================================================================

class TestQuantitativeObjective:
    def test_maximises_single_agent(self):
        territory = Territory.from_ascii(["..", ".."])
        opt, v = _build_full_model(territory, num_agents=2, horizon=2)
        quantitative_objective(opt, v, agent_id=0)
        assert opt.check() == z3.sat
        m = opt.model()
        p0 = _model_val(m, v.payoff[0])
        assert p0 >= 2

    def test_maximise_agent_1(self):
        """Maximising agent 1 should give agent 1 at least as good as any other."""
        territory = Territory.from_ascii(["..", ".."])
        opt, v = _build_full_model(territory, num_agents=2, horizon=2)
        quantitative_objective(opt, v, agent_id=1)
        assert opt.check() == z3.sat
        m = opt.model()
        p1 = _model_val(m, v.payoff[1])
        assert p1 >= 2

    def test_invalid_agent_raises(self):
        territory = Territory.from_ascii([".."])
        opt, v = _build_full_model(territory, num_agents=1, horizon=1)
        with pytest.raises(ValueError, match="out of range"):
            quantitative_objective(opt, v, agent_id=5)

    def test_negative_agent_raises(self):
        territory = Territory.from_ascii([".."])
        opt, v = _build_full_model(territory, num_agents=1, horizon=1)
        with pytest.raises(ValueError, match="out of range"):
            quantitative_objective(opt, v, agent_id=-1)

    def test_single_agent_trivial(self):
        """1 agent, 1 sector, 1 round: payoff must be 1."""
        territory = Territory.from_ascii(["."])
        opt, v = _build_full_model(territory, num_agents=1, horizon=1)
        quantitative_objective(opt, v, agent_id=0)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.payoff[0]) == 1
