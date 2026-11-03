"""Tests for smt_objectives.py (thesis Definition 4.6.1 and Definition 4.7.1).

Oracle: every control vector in which each agent controls exactly ``m``
sectors is the final state of some play, so the optima over plays are the
optima over these vectors.
"""

from __future__ import annotations

import pytest

from rcmas.model import Territory, rewards_from_owner
from rcmas.smt_objectives import collective_objective, individual_objective
from rcmas.smt_solve import build_base_encoding

from .brute_force import final_states

z3 = pytest.importorskip("z3")


def _optimise(t: Territory, n: int, add_objective, demands=None):
    m = len(t) // n
    base = build_base_encoding(territory=t, num_agents=n, horizon=m, demands=demands)
    opt = z3.Optimize()
    opt.add(*base.constraints)
    add_objective(opt, base.variables)
    assert opt.check() == z3.sat
    mdl = opt.model()
    final = tuple(mdl.eval(base.variables.owner[i][m]).as_long() for i in range(len(t)))
    encoded = tuple(mdl.eval(r).as_long() for r in base.variables.reward)
    return final, encoded


GRIDS = [
    (["..", ".."], 2),
    (["...", "..."], 2),
    (["...", "..."], 3),
    (["...", "...", "..."], 2),
    (["..#", "...", ".#."], 2),
]


class TestCollectiveObjective:
    @pytest.mark.parametrize("grid,n", GRIDS)
    def test_maximum_equals_brute_force(self, grid, n):
        """Theorem 4.6.1: the optimum of f is the collective optimum."""
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        best = max(
            sum(rewards_from_owner(s, n, nb))
            for s in final_states(len(t), n, len(t) // n)
        )
        final, encoded = _optimise(t, n, collective_objective)
        assert sum(encoded) == best
        assert sum(rewards_from_owner(final, n, nb)) == best

    def test_with_demands(self):
        """With d = (3, 1) on 2x3 the optimum is still 6 (3 + 3)."""
        t = Territory.from_ascii(["...", "..."])
        final, encoded = _optimise(t, 2, collective_objective, demands=(3, 1))
        assert sum(encoded) == 6
        assert rewards_from_owner(final, 2, t.neighbor_map())[0] >= 3


class TestIndividualObjective:
    @pytest.mark.parametrize("grid,n", GRIDS)
    def test_maximum_equals_brute_force(self, grid, n):
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        m = len(t) // n
        best = max(
            rewards_from_owner(s, n, nb)[n - 1] for s in final_states(len(t), n, m)
        )
        final, encoded = _optimise(
            t, n, lambda opt, v: individual_objective(opt, v, n - 1)
        )
        assert encoded[n - 1] == best
        assert rewards_from_owner(final, n, nb)[n - 1] == best

    def test_invalid_agent_raises(self):
        t = Territory.from_ascii(["..", ".."])
        base = build_base_encoding(territory=t, num_agents=2, horizon=2)
        with pytest.raises(ValueError):
            individual_objective(z3.Optimize(), base.variables, 2)
        with pytest.raises(ValueError):
            individual_objective(z3.Optimize(), base.variables, -1)
