from __future__ import annotations

from rcmas_core.engine import Coord, Territory

from rcmas_testbeds.testbeds.smt_co import solve_smt_game


def test_fixed_policy_agents_not_optimized() -> None:
    # Single-sector game: if agent[1] is fixed to claim the only cell,
    # then maximizing agent[0]'s payoff cannot change that.
    territory = Territory.from_ascii(["."])

    # State key is the full ownership vector in sector order.
    initial_state = (-1,)
    fixed_policy_agent_1 = {initial_state: Coord(0, 0)}

    fixed = solve_smt_game(
        territory=territory,
        num_agents=2,
        horizon=1,
        objective=0,
        fixed_policy_by_agent=(None, fixed_policy_agent_1),
        require_victory=True,
        debug=True,
    )
    assert fixed.is_sat
    assert fixed.actions_by_round is not None
    assert fixed.payoff_by_agent is not None

    assert fixed.actions_by_round[0][1] == Coord(0, 0)
    assert fixed.payoff_by_agent[0] == 0

    # Without the fixed policy, optimizing agent[0] should claim the cell.
    free = solve_smt_game(
        territory=territory,
        num_agents=2,
        horizon=1,
        objective=0,
        fixed_policy_by_agent=None,
        require_victory=True,
        debug=True,
    )
    assert free.is_sat
    assert free.actions_by_round is not None
    assert free.payoff_by_agent is not None

    assert free.actions_by_round[0][0] == Coord(0, 0)
    assert free.payoff_by_agent[0] == 1
