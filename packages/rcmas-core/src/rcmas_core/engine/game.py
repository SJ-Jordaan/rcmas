from __future__ import annotations

from dataclasses import dataclass

from .outcome import GameOutcome, GameStatus
from .state import GameState


@dataclass(slots=True)
class GameResult:
    final_state: GameState
    outcome: GameOutcome

    @property
    def scores(self) -> tuple[int, ...]:
        return self.final_state.scores()


class GameEngine:
    """Runs a game given agents (I/O-free)."""

    def __init__(self, initial_state: GameState):
        self._state = initial_state

    @property
    def state(self) -> GameState:
        return self._state

    def step(self, actions: dict[int, object]) -> GameOutcome:
        # Actions are expected to be Coord|None; kept loose here to avoid import cycles.
        state, outcome = self._state.step(actions)  # type: ignore[arg-type]
        self._state = state
        return outcome

    def run(self, agents: list[object], *, max_rounds: int | None = None) -> GameResult:
        if len(agents) != self._state.num_agents:
            raise ValueError("agents length must match num_agents")

        rounds = 0
        while True:
            if max_rounds is not None and rounds >= max_rounds:
                return GameResult(self._state, GameOutcome(GameStatus.DEFEAT, "max_rounds"))

            if self._state.is_terminal():
                return GameResult(self._state, GameOutcome(GameStatus.VICTORY, "all sectors acquired"))

            joint_actions: dict[int, object] = {}
            for agent_id, agent in enumerate(agents):
                act = agent.select_action(self._state, agent_id)  # type: ignore[attr-defined]
                joint_actions[agent_id] = act

            outcome = self.step(joint_actions)
            rounds += 1

            if outcome.status != GameStatus.ONGOING:
                return GameResult(self._state, outcome)
