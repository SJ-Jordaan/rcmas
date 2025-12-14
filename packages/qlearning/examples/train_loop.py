from __future__ import annotations

import random
from pathlib import Path

from qlearning.engine import GameEngine, GameState, GameStatus, Territory
from qlearning.rl.encoding import TabularEncoding
from qlearning.rl.qtable import QTable


def load_territory(path: Path) -> Territory:
    return Territory.from_ascii(path.read_text(encoding="utf-8").splitlines())


def epsilon_schedule(ep: int, *, start: float = 1.0, end: float = 0.05, decay_episodes: int = 800) -> float:
    if decay_episodes <= 0:
        return end
    t = min(1.0, ep / decay_episodes)
    return start + t * (end - start)


def greedy_action(state: GameState, encoding: TabularEncoding, qtable: QTable, rng: random.Random) -> object:
    options = list(state.available_sectors())
    if not options:
        return None

    sk = encoding.state_key(state)
    best = None
    best_v = None
    for c in options:
        ak = encoding.action_key(state, c)
        v = qtable.get(sk, ak)
        if best is None or v > best_v:  # type: ignore[operator]
            best, best_v = c, v

    # Small tie-break randomness keeps things moving when values are all 0.
    if best_v == 0.0:
        return rng.choice(options)
    return best


def eval_episode(territory: Territory, qtables: list[QTable], *, seed: int = 0) -> tuple[str, tuple[int, ...]]:
    rng = random.Random(seed)
    state = GameState.new(territory, num_agents=len(qtables))
    engine = GameEngine(state)
    enc = TabularEncoding(num_agents=len(qtables))

    class _Policy:
        def __init__(self, qt: QTable):
            self.qt = qt

        def select_action(self, st: GameState, agent_id: int):
            return greedy_action(st, enc, self.qt, rng)

    agents = [_Policy(qt) for qt in qtables]
    result = engine.run(agents)
    return result.outcome.status.value, result.scores


def main() -> int:
    # In the monorepo layout, grids live at the repo top-level.
    grid_path = Path(__file__).resolve().parents[2] / "grids" / "example.txt"
    territory = load_territory(grid_path)

    num_agents = 2
    episodes = 500000
    alpha = 0.3
    gamma = 0.95
    defeat_penalty = 50.0

    rng = random.Random(0)
    encoding = TabularEncoding(num_agents=num_agents)
    qtables = [QTable.empty() for _ in range(num_agents)]

    for ep in range(episodes):
        eps = epsilon_schedule(ep)
        state = GameState.new(territory, num_agents=num_agents)

        while True:
            if state.is_terminal():
                break

            sk = encoding.state_key(state)

            # epsilon-greedy joint action
            actions = {}
            action_keys = {}
            for agent_id in range(num_agents):
                options = list(state.available_sectors())
                if not options:
                    actions[agent_id] = None
                    continue
                if rng.random() < eps:
                    act = rng.choice(options)
                else:
                    act = greedy_action(state, encoding, qtables[agent_id], rng)
                actions[agent_id] = act
                if act is not None:
                    action_keys[agent_id] = encoding.action_key(state, act)

            next_state, outcome = state.step(actions)
            terminal = outcome.status != GameStatus.ONGOING

            # rewards
            if outcome.status == GameStatus.DEFEAT:
                rewards = [-defeat_penalty] * num_agents
            elif terminal:
                rewards = list(map(float, next_state.scores()))
            else:
                rewards = [0.0] * num_agents

            # Q updates
            for agent_id in range(num_agents):
                ak = action_keys.get(agent_id)
                if ak is None:
                    continue

                if terminal:
                    target = rewards[agent_id]
                else:
                    next_sk = encoding.state_key(next_state)
                    next_opts = list(next_state.available_sectors())
                    next_aks = [encoding.action_key(next_state, c) for c in next_opts]
                    best_next = qtables[agent_id].best_action(next_sk, next_aks)
                    next_max = 0.0 if best_next is None else qtables[agent_id].get(next_sk, best_next)
                    target = rewards[agent_id] + gamma * next_max

                old = qtables[agent_id].get(sk, ak)
                qtables[agent_id].set(sk, ak, (1 - alpha) * old + alpha * target)

            state = next_state
            if terminal:
                break

        if (ep + 1) % 150 == 0:
            status, scores = eval_episode(territory, qtables, seed=ep)
            print(f"ep={ep+1} eps={eps:.3f} eval_status={status} eval_scores={scores}")

    out_dir = Path(__file__).resolve().parents[1] / "q_tables" / "example"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, qt in enumerate(qtables):
        qt.save_json(out_dir / f"agent_{i}.json")
    print(f"saved demo qtables -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
