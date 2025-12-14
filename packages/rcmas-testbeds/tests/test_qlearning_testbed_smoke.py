from __future__ import annotations

import pytest

from rcmas_core.engine import GameEngine, GameState, Territory


def test_qlearning_testbed_smoke_trains_and_runs() -> None:
    qlearning = pytest.importorskip("qlearning")  # noqa: F841

    from rcmas_testbeds.testbeds.qlearning_tb import QLearningTestbed

    territory = Territory.from_ascii(["..", ".."])  # 2x2

    tb = QLearningTestbed(train_episodes=10, seed=0, eval_epsilon=0.0)
    agents = tb.build_agents(territory=territory, num_agents=2, max_rounds=2)

    engine = GameEngine(GameState.new(territory, num_agents=2))
    result = engine.run(agents, max_rounds=2)

    assert result.final_state.round_index <= 2
    assert len(result.scores) == 2
