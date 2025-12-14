from rcmas_core.engine import Coord, GameEngine, GameState, Territory


def test_engine_smoke_runs_to_terminal() -> None:
    territory = Territory.from_ascii(
        [
            "..",
            "..",
        ]
    )
    state = GameState.new(territory, num_agents=2)

    class Agent:
        def __init__(self, moves: list[Coord | None]):
            self._moves = moves

        def select_action(self, state: GameState, agent_id: int) -> Coord | None:  # noqa: ARG002
            return self._moves.pop(0) if self._moves else None

    a0 = Agent([Coord(0, 0), Coord(1, 0)])
    a1 = Agent([Coord(0, 1), Coord(1, 1)])

    engine = GameEngine(state)
    result = engine.run([a0, a1], max_rounds=10)

    assert result.final_state.is_terminal()
    assert result.scores == (2, 2)
