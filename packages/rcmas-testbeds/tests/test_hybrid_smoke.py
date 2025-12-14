from __future__ import annotations

from rcmas_core.engine import Territory

from rcmas_testbeds.testbeds.hybrid import HybridConfig, solve_hybrid_ne


def test_hybrid_smoke_2x2_h2_rl_disabled() -> None:
    territory = Territory.from_ascii(["..", ".."])  # 4 sectors

    res = solve_hybrid_ne(
        territory=territory,
        num_agents=2,
        horizon=2,
        cfg=HybridConfig(
            max_iters=5,
            rl_episodes_per_iter=0,  # keep test deterministic and fast
            timeout_ms=2000,
        ),
    )

    assert res.is_sat
    assert res.final_solution is not None
    assert res.final_solution.final_state is not None
    assert all(o != -1 for o in res.final_solution.final_state.owner_by_index)


def test_hybrid_smoke_2x2_h2_rl_enabled_topk() -> None:
    territory = Territory.from_ascii(["..", ".."])  # 4 sectors

    res = solve_hybrid_ne(
        territory=territory,
        num_agents=2,
        horizon=2,
        cfg=HybridConfig(
            max_iters=5,
            rl_episodes_per_iter=20,
            rl_top_k_actions=2,
            timeout_ms=5000,
            seed=0,
        ),
    )

    assert res.is_sat
    assert res.final_solution is not None
    assert res.final_solution.final_state is not None
    assert all(o != -1 for o in res.final_solution.final_state.owner_by_index)
