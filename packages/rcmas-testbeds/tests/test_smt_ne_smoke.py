from __future__ import annotations

from rcmas_core.engine import Territory

from rcmas_testbeds.testbeds.smt_ne import solve_naive_ne


def test_smt_ne_smoke_2x2_h2() -> None:
    territory = Territory.from_ascii(["..", ".."])  # 4 sectors

    res = solve_naive_ne(territory=territory, num_agents=2, horizon=2, seed=0, max_iters=5)
    assert res.is_sat
    assert res.strategy is not None
    assert res.payoff_by_agent is not None
    assert len(res.payoff_by_agent) == 2

    # Ensure the final fixed-strategy solve is a victory (all sectors claimed) under our model.
    assert res.final_solution is not None
    assert res.final_solution.final_state is not None
    assert all(o != -1 for o in res.final_solution.final_state.owner_by_index)
