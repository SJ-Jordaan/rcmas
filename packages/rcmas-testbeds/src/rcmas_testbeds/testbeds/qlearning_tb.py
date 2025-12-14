from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from rcmas_core.engine import Territory

from .base import Testbed


@dataclass(slots=True)
class _EvalQAgent:
    """Evaluation agent for a learned Q-table.

    This intentionally uses *stochastic tie-breaking* (and optional epsilon) to
    reduce symmetry-driven collisions that happen when both agents pick the same
    greedy argmax.
    """

    qtable: "QTable"
    num_agents: int
    epsilon: float
    rng: random.Random

    def select_action(self, state: "GameState", agent_id: int):
        from qlearning.rl.encoding import TabularEncoding

        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError("agent_id out of range")
        if state.num_agents != self.num_agents:
            raise ValueError("num_agents mismatch")

        options = list(state.available_sectors())
        if not options:
            return None

        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return self.rng.choice(options)

        # Randomize option order so QTable.best_action breaks ties stochastically.
        self.rng.shuffle(options)
        enc = TabularEncoding(num_agents=self.num_agents)
        sk = enc.state_key(state)
        action_keys = [enc.action_key(state, c) for c in options]
        best_ak = self.qtable.best_action(sk, action_keys)
        if best_ak is None:
            return self.rng.choice(options)
        return enc.decode_action(state, best_ak)


@dataclass(frozen=True, slots=True)
class QLearningTestbed(Testbed):
    """Pure Q-learning baseline using the `qlearning` package.

    Trains independent Q-learners in self-play for a fixed number of episodes,
    then evaluates greedily via `GameEngine`.
    """

    name: str = "qlearning"
    train_episodes: int = 2000
    seed: int = 0
    eval_epsilon: float = 0.02

    def build_agents(self, *, territory: Territory, num_agents: int, max_rounds: int) -> list[object]:
        try:
            from qlearning.rl.trainer import SelfPlayTrainer, TrainConfig
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "qlearning package is required for the qlearning testbed; "
                "install it editable (e.g. `pip install -e ../qlearning`)"
            ) from e

        if self.train_episodes <= 0:
            raise ValueError("train_episodes must be >= 1")
        if num_agents <= 0:
            raise ValueError("num_agents must be >= 1")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be >= 1")

        # Trainer writes artifacts; keep them in a temp directory by default.
        with TemporaryDirectory(prefix="rcmas-qlearning-") as tmp:
            trainer = SelfPlayTrainer(out_dir=Path(tmp))
            cfg = TrainConfig(episodes=self.train_episodes, max_rounds=max_rounds)
            artifacts = trainer.train(territory, num_agents, cfg, seed=self.seed)

            agents: list[object] = []
            for i, qt in enumerate(artifacts.qtables):
                agents.append(
                    _EvalQAgent(
                        qtable=qt,
                        num_agents=num_agents,
                        epsilon=self.eval_epsilon,
                        rng=random.Random(self.seed + 10_000 + i),
                    )
                )
            return agents
