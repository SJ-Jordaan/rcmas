from __future__ import annotations

import json
import random
from dataclasses import dataclass
import hashlib
from pathlib import Path

from qlearning.engine import Coord, GameOutcome, GameStatus, GameState, Territory

from .encoding import TabularEncoding
from .qlearning import QLearningPolicy
from .qtable import QTable


def territory_id(territory: Territory) -> str:
    # Stable across runs for a given set of sectors
    sectors = territory.ordered_sectors()
    h = hashlib.sha1()
    for c in sectors:
        h.update(f"{c.x},{c.y};".encode("utf-8"))
    return "t" + h.hexdigest()[:12]


@dataclass(slots=True)
class TrainConfig:
    episodes: int
    max_rounds: int | None = None
    alpha: float = 0.3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    defeat_penalty: float = 1000.0
    step_success_reward: float = 1.0
    delta_region_reward: float = 2.0
    terminal_score_reward: float = 1.0


@dataclass(slots=True)
class TrainingArtifacts:
    directory: Path
    qtables: list[QTable]


class SelfPlayTrainer:
    """Independent Q-learning in self-play.

    This is not guaranteed to converge in general-sum multi-agent games, but it is
    a straightforward baseline that can learn reasonable strategies on small grids.
    """

    def __init__(self, *, out_dir: Path):
        self._out_dir = out_dir

    def train(self, territory: Territory, num_agents: int, cfg: TrainConfig, *, seed: int = 0) -> TrainingArtifacts:
        rngs = [random.Random(seed + i) for i in range(num_agents)]
        encoding = TabularEncoding(num_agents=num_agents)
        qtables = [QTable.empty() for _ in range(num_agents)]

        for episode in range(cfg.episodes):
            eps = self._epsilon_for_episode(cfg, episode)
            policies = [
                QLearningPolicy(qtable=qtables[i], encoding=encoding, epsilon=eps, rng=rngs[i])
                for i in range(num_agents)
            ]

            state = GameState.new(territory, num_agents)
            rounds = 0

            while True:
                if cfg.max_rounds is not None and rounds >= cfg.max_rounds:
                    outcome = GameOutcome(GameStatus.DEFEAT, "max_rounds")
                    self._apply_terminal_updates(state, qtables, encoding, {}, outcome, cfg)
                    break

                if state.is_terminal():
                    break

                skeys = [encoding.state_key(state) for _ in range(num_agents)]
                actions: dict[int, Coord | None] = {}
                action_keys: dict[int, str] = {}

                for agent_id in range(num_agents):
                    act = policies[agent_id].select_action(state)
                    actions[agent_id] = act
                    if act is not None:
                        action_keys[agent_id] = encoding.action_key(state, act)

                next_state, outcome = state.step(actions)

                terminal = outcome.status != GameStatus.ONGOING
                rewards = self._step_rewards(state, actions, next_state, outcome, cfg, num_agents)

                for agent_id in range(num_agents):
                    if agent_id not in action_keys:
                        continue
                    s = skeys[agent_id]
                    a = action_keys[agent_id]
                    r = rewards[agent_id]

                    if terminal:
                        target = r
                    else:
                        next_sk = encoding.state_key(next_state)
                        next_opts = list(next_state.available_sectors())
                        next_action_keys = [encoding.action_key(next_state, c) for c in next_opts]
                        best_next = qtables[agent_id].best_action(next_sk, next_action_keys)
                        next_max = 0.0 if best_next is None else qtables[agent_id].get(next_sk, best_next)
                        target = r + cfg.gamma * next_max

                    old = qtables[agent_id].get(s, a)
                    new = (1.0 - cfg.alpha) * old + cfg.alpha * target
                    qtables[agent_id].set(s, a, new)

                state = next_state
                rounds += 1

                if terminal:
                    break

        tid = territory_id(territory)
        out = self._out_dir / tid
        out.mkdir(parents=True, exist_ok=True)

        meta = {
            "territory_id": tid,
            "num_agents": num_agents,
            "num_sectors": len(territory),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        for i, qt in enumerate(qtables):
            qt.save_json(out / f"agent_{i}.json")

        return TrainingArtifacts(directory=out, qtables=qtables)

    def _epsilon_for_episode(self, cfg: TrainConfig, episode: int) -> float:
        if cfg.epsilon_decay_episodes <= 0:
            return cfg.epsilon_end
        t = min(1.0, episode / cfg.epsilon_decay_episodes)
        return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)

    def _step_rewards(
        self,
        prev_state: GameState,
        actions: dict[int, Coord | None],
        next_state: GameState,
        outcome: GameOutcome,
        cfg: TrainConfig,
        num_agents: int,
    ) -> list[float]:
        # Safety-critical: any collision is a big negative for everyone.
        if outcome.status == GameStatus.DEFEAT:
            return [-cfg.defeat_penalty for _ in range(num_agents)]

        rewards = [0.0 for _ in range(num_agents)]

        # Dense shaping: reward each successful claim and progress toward larger cohesive regions.
        prev_sizes = [prev_state.largest_region_size(i) for i in range(num_agents)]
        next_sizes = [next_state.largest_region_size(i) for i in range(num_agents)]

        for agent_id in range(num_agents):
            act = actions.get(agent_id)
            if act is not None:
                # If we didn't lose, any action implies a successful, unique claim.
                rewards[agent_id] += cfg.step_success_reward

            delta = next_sizes[agent_id] - prev_sizes[agent_id]
            if delta:
                rewards[agent_id] += cfg.delta_region_reward * float(delta)

        # Terminal: also pay out final score (scaled) to reinforce overall objective.
        if outcome.status == GameStatus.VICTORY:
            scores = next_state.scores()
            for i in range(num_agents):
                rewards[i] += cfg.terminal_score_reward * float(scores[i])

        return rewards

    def _apply_terminal_updates(
        self,
        state: GameState,
        qtables: list[QTable],
        encoding: TabularEncoding,
        action_keys: dict[int, str],
        outcome: GameOutcome,
        cfg: TrainConfig,
    ) -> None:
        # Backwards-compat helper (unused by main loop today).
        rewards = [-cfg.defeat_penalty for _ in range(state.num_agents)] if outcome.status == GameStatus.DEFEAT else [0.0 for _ in range(state.num_agents)]
        for agent_id, ak in action_keys.items():
            sk = encoding.state_key(state)
            old = qtables[agent_id].get(sk, ak)
            new = (1.0 - cfg.alpha) * old + cfg.alpha * rewards[agent_id]
            qtables[agent_id].set(sk, ak, new)
