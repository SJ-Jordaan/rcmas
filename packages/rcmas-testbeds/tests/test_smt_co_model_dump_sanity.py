from __future__ import annotations

from rcmas_core.engine import Territory

from rcmas_testbeds.testing.smt_co_sanity import assert_owner_history_monotone, assert_smt_co_debug_sanity
from rcmas_testbeds.testbeds.smt_co import solve_collective_optimality


def test_smt_co_debug_variables_make_sense_4x4_h8() -> None:
    territory = Territory.from_ascii(["....", "....", "....", "...."])
    sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=8, debug=True)

    assert_smt_co_debug_sanity(
        sol=sol,
        territory=territory,
        num_agents=2,
        expected_payoff_by_agent=(8, 8),
        require_terminal=True,
    )
    assert_owner_history_monotone(sol)
