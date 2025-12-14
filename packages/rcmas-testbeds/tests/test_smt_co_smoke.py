from rcmas_core.engine import GameState, Territory
from rcmas_testbeds.testbeds.registry import build_testbed


def test_smt_co_reaches_terminal_on_tiny_grid() -> None:
    territory = Territory.from_ascii(
        [
            "..",
            "..",
        ]
    )
    tb = build_testbed("smt-co")

    sol = tb.solve(territory=territory, num_agents=2, horizon=2)  # type: ignore[attr-defined]
    assert sol.is_sat
    assert sol.final_state is not None
    assert sol.final_state.is_terminal()
    assert len(sol.final_state.scores()) == 2
