from __future__ import annotations

import io

import pytest

from rcmas_core.engine import Territory

from rcmas_testbeds.testing.smt_co_sanity import assert_owner_history_monotone, assert_smt_co_debug_sanity
from rcmas_testbeds.testbeds.smt_co import solve_collective_optimality


@pytest.mark.parametrize(
    ("grid", "agents", "horizon", "expected_payoff", "require_terminal"),
    [
        # Rectangular baseline
        ("....\n....\n....\n....\n", 2, 8, (8, 8), True),
        # Irregular / non-rectangular with holes
        ("..#.\n.#..\n#.#\n...#\n", 2, 5, None, False),
    ],
)
def test_smt_co_debug_sanity_generic(
    grid: str,
    agents: int,
    horizon: int,
    expected_payoff: tuple[int, ...] | None,
    require_terminal: bool,
) -> None:
    territory = Territory.from_ascii(io.StringIO(grid))
    sol = solve_collective_optimality(territory=territory, num_agents=agents, horizon=horizon, debug=True)

    assert_smt_co_debug_sanity(
        sol=sol,
        territory=territory,
        num_agents=agents,
        expected_payoff_by_agent=expected_payoff,
        require_terminal=require_terminal,
    )
    assert_owner_history_monotone(sol)
