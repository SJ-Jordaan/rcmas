"""Section 5.2: Q-learning for RCMAS.

Provides tabular Q-learning with state/action encoding, a Q-table,
and a best-response training function used by Q-IBIS.
Also includes the self-play trainer for standalone Q-learning experiments.
"""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .model import Coord, State, Territory, CollisionError, evolve, largest_region_size, scores


# ---------------------------------------------------------------------------
# State / action encoding
# ---------------------------------------------------------------------------

def state_key(owner_by_index: tuple[int, ...]) -> str:
    """Encode an ownership vector as a URL-safe base64 string."""
    raw = bytes((o + 1) if o >= 0 else 0 for o in owner_by_index)
    return base64.urlsafe_b64encode(raw).decode("ascii")


def action_key(state: State, action: Coord) -> str:
    """Encode an action as the sector index string."""
    idx = state.territory.index_of(action)
    if idx is None:
        raise ValueError("action not in territory")
    return str(idx)


def decode_action(state: State, ak: str) -> Coord:
    """Decode an action key back to a Coord."""
    return state.ordered_sectors[int(ak)]


# ---------------------------------------------------------------------------
# Q-table
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class QTable:
    """Simple tabular Q(s,a) storage with JSON serialization."""

    q: dict[str, dict[str, float]]

    @staticmethod
    def empty() -> QTable:
        return QTable(q={})

    def get(self, sk: str, ak: str) -> float:
        return self.q.get(sk, {}).get(ak, 0.0)

    def set(self, sk: str, ak: str, value: float) -> None:
        self.q.setdefault(sk, {})[ak] = float(value)

    def best_action(self, sk: str, action_keys: list[str]) -> str | None:
        if not action_keys:
            return None
        best = action_keys[0]
        best_v = self.get(sk, best)
        for ak in action_keys[1:]:
            v = self.get(sk, ak)
            if v > best_v:
                best, best_v = ak, v
        return best

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.q, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load_json(path: Path) -> QTable:
        return QTable(q=json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Q-value helpers
# ---------------------------------------------------------------------------

def best_q_action(
    q: dict[str, dict[str, float]], sk: str, action_keys: list[str],
) -> str | None:
    """Return the action key with the highest Q-value."""
    if not action_keys:
        return None
    best = action_keys[0]
    best_v = q.get(sk, {}).get(best, 0.0)
    for ak in action_keys[1:]:
        v = q.get(sk, {}).get(ak, 0.0)
        if v > best_v:
            best, best_v = ak, v
    return best


def top_k_actions(
    q: dict[str, dict[str, float]], sk: str, action_keys: list[str], *, k: int,
) -> list[str]:
    """Return the top-k action keys by Q-value (descending)."""
    if not action_keys or k <= 0:
        return []
    scored = [(q.get(sk, {}).get(ak, 0.0), ak) for ak in action_keys]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [ak for _, ak in scored[:min(k, len(scored))]]


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward-shaping parameters for Q-learning."""

    defeat_penalty: float = 1000.0
    step_success_reward: float = 1.0
    delta_region_reward: float = 2.0
    terminal_score_reward: float = 1.0


# ---------------------------------------------------------------------------
# Self-play trainer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrainConfig:
    """Training hyperparameters for self-play Q-learning."""

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


def train_self_play(
    territory: Territory,
    num_agents: int,
    cfg: TrainConfig,
    *,
    seed: int = 0,
    out_dir: Path | None = None,
) -> TrainingArtifacts:
    """Train independent Q-learners in self-play."""
    rngs = [random.Random(seed + i) for i in range(num_agents)]
    qtables = [QTable.empty() for _ in range(num_agents)]

    for episode in range(cfg.episodes):
        eps = _epsilon(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_episodes, episode)
        state = State.initial(territory, num_agents)
        rounds = 0

        while True:
            if cfg.max_rounds is not None and rounds >= cfg.max_rounds:
                break
            if state.is_terminal():
                break

            skeys = [state_key(state.owner_by_index)] * num_agents
            actions: dict[int, Coord | None] = {}
            akeys: dict[int, str] = {}

            for aid in range(num_agents):
                act = _epsilon_greedy_action(state, qtables[aid], rngs[aid], eps)
                actions[aid] = act
                if act is not None:
                    akeys[aid] = action_key(state, act)

            try:
                next_state = evolve(state, actions)
                defeated = False
            except CollisionError:
                defeated = True
                next_state = state

            terminal = defeated or next_state.is_terminal()
            rewards = _step_rewards(state, actions, next_state, defeated, num_agents, cfg)

            for aid in range(num_agents):
                if aid not in akeys:
                    continue
                _q_update(
                    qtables[aid], skeys[aid], akeys[aid], rewards[aid],
                    next_state, terminal, cfg.alpha, cfg.gamma,
                )

            state = next_state
            rounds += 1
            if terminal:
                break

    # Save artifacts
    if out_dir is None:
        from tempfile import mkdtemp
        out_dir = Path(mkdtemp(prefix="rcmas-qlearning-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, qt in enumerate(qtables):
        qt.save_json(out_dir / f"agent_{i}.json")

    return TrainingArtifacts(directory=out_dir, qtables=qtables)


def _epsilon(start: float, end: float, decay: int, episode: int) -> float:
    if decay <= 0:
        return end
    t = min(1.0, episode / decay)
    return start + t * (end - start)


def _epsilon_greedy_action(
    state: State, qtable: QTable, rng: random.Random, eps: float,
) -> Coord | None:
    options = list(state.available_actions())
    if not options:
        return None
    if rng.random() < eps:
        return rng.choice(options)
    sk = state_key(state.owner_by_index)
    aks = [action_key(state, c) for c in options]
    best = qtable.best_action(sk, aks)
    if best is None:
        return rng.choice(options)
    return decode_action(state, best)


def _step_rewards(
    prev: State,
    actions: dict[int, Coord | None],
    next_state: State,
    defeated: bool,
    num_agents: int,
    cfg: TrainConfig,
) -> list[float]:
    if defeated:
        return [-cfg.defeat_penalty] * num_agents

    rewards = [0.0] * num_agents
    prev_sizes = [largest_region_size(prev, i) for i in range(num_agents)]
    next_sizes = [largest_region_size(next_state, i) for i in range(num_agents)]

    for aid in range(num_agents):
        if actions.get(aid) is not None:
            rewards[aid] += cfg.step_success_reward
        delta = next_sizes[aid] - prev_sizes[aid]
        if delta:
            rewards[aid] += cfg.delta_region_reward * float(delta)

    if next_state.is_terminal():
        sc = scores(next_state)
        for i in range(num_agents):
            rewards[i] += cfg.terminal_score_reward * float(sc[i])

    return rewards


def _q_update(
    qtable: QTable,
    sk: str,
    ak: str,
    reward: float,
    next_state: State,
    terminal: bool,
    alpha: float,
    gamma: float,
) -> None:
    if terminal:
        target = reward
    else:
        next_sk = state_key(next_state.owner_by_index)
        next_opts = list(next_state.available_actions())
        next_aks = [action_key(next_state, c) for c in next_opts]
        best_next = qtable.best_action(next_sk, next_aks)
        next_max = 0.0 if best_next is None else qtable.get(next_sk, best_next)
        target = reward + gamma * next_max

    old = qtable.get(sk, ak)
    new = (1.0 - alpha) * old + alpha * float(target)
    qtable.set(sk, ak, new)
