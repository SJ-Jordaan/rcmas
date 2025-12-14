from __future__ import annotations

import argparse
from pathlib import Path

from qlearning.agents import GreedyLargestRegionAgent, QTableAgent, RandomAgent
from qlearning.engine import Coord, GameEngine, GameState, Territory
from qlearning.rl import SelfPlayTrainer, TrainConfig


def _load_territory(path: Path) -> Territory:
    return Territory.from_ascii(path.read_text(encoding="utf-8").splitlines())


def _load_grid_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qlearning")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="Quick import/sanity check")

    sim = sub.add_parser("simulate", help="Run a single game")
    sim.add_argument("--grid", type=Path, required=True, help="Path to ASCII grid ('.' = sector)")
    sim.add_argument(
        "--agents",
        nargs="+",
        default=["random", "random"],
        help="random|greedy|qtable:/path/to/agent.json per agent",
    )
    sim.add_argument("--max-rounds", type=int, default=None)
    sim.add_argument("--render", action="store_true", help="Print final board")
    sim.add_argument("--epsilon", type=float, default=0.0, help="Exploration for qtable:* agents")
    sim.add_argument(
        "--stochastic-ties",
        action="store_true",
        help="If set, qtable:* agents break Q-value ties randomly",
    )

    tr = sub.add_parser("train", help="Train Q-tables on a fixed grid")
    tr.add_argument("--grid", type=Path, required=True, help="Path to ASCII grid ('.' = sector)")
    tr.add_argument("--num-agents", type=int, default=2)
    tr.add_argument("--episodes", type=int, default=1000)
    tr.add_argument("--out-dir", type=Path, default=Path("q_tables"))
    tr.add_argument("--alpha", type=float, default=0.3)
    tr.add_argument("--gamma", type=float, default=0.95)
    tr.add_argument("--epsilon-start", type=float, default=1.0)
    tr.add_argument("--epsilon-end", type=float, default=0.05)
    tr.add_argument("--epsilon-decay-episodes", type=int, default=500)
    tr.add_argument("--defeat-penalty", type=float, default=1000.0)
    tr.add_argument("--step-success-reward", type=float, default=1.0)
    tr.add_argument("--delta-region-reward", type=float, default=2.0)
    tr.add_argument("--terminal-score-reward", type=float, default=1.0)
    tr.add_argument("--max-rounds", type=int, default=None)
    tr.add_argument("--seed", type=int, default=0)

    return parser


def _make_agent(spec: str, *, num_agents: int, seed: int, epsilon: float, deterministic_ties: bool):
    spec = spec.strip()
    low = spec.lower()
    if low == "random":
        return RandomAgent()
    if low == "greedy":
        return GreedyLargestRegionAgent()
    if low.startswith("qtable:"):
        path = Path(spec.split(":", 1)[1]).expanduser()
        return QTableAgent.load_json(
            path,
            num_agents=num_agents,
            epsilon=epsilon,
            seed=seed,
            deterministic_ties=deterministic_ties,
        )
    raise ValueError(f"unknown agent: {spec}")


def _agent_char(agent_id: int) -> str:
    if 0 <= agent_id < 26:
        return chr(ord("A") + agent_id)
    return str(agent_id % 10)


def _render_final(grid_lines: list[str], state: GameState) -> str:
    rows: list[str] = []
    for y, line in enumerate(grid_lines):
        out = list(line)
        for x, ch in enumerate(out):
            if ch != ".":
                continue
            owner = state.owner_of(Coord(x, y))
            out[x] = "." if owner is None else _agent_char(owner)
        rows.append("".join(out))
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "smoke":
        print("ok")
        return 0

    if args.cmd == "simulate":
        grid_lines = _load_grid_lines(args.grid)
        territory = Territory.from_ascii(grid_lines)
        num_agents = len(args.agents)
        deterministic_ties = not args.stochastic_ties
        agents = [
            _make_agent(
                n,
                num_agents=num_agents,
                seed=i,
                epsilon=args.epsilon,
                deterministic_ties=deterministic_ties,
            )
            for i, n in enumerate(args.agents)
        ]
        state = GameState.new(territory, len(agents))
        engine = GameEngine(state)
        result = engine.run(agents, max_rounds=args.max_rounds)
        print(f"status={result.outcome.status} reason={result.outcome.reason}")
        print(f"scores={result.scores}")
        if args.render:
            print("\nfinal:")
            print(_render_final(grid_lines, result.final_state))
        return 0

    if args.cmd == "train":
        territory = _load_territory(args.grid)
        cfg = TrainConfig(
            episodes=args.episodes,
            max_rounds=args.max_rounds,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
            defeat_penalty=args.defeat_penalty,
            step_success_reward=args.step_success_reward,
            delta_region_reward=args.delta_region_reward,
            terminal_score_reward=args.terminal_score_reward,
        )
        trainer = SelfPlayTrainer(out_dir=args.out_dir)
        artifacts = trainer.train(territory, args.num_agents, cfg, seed=args.seed)
        print(f"saved={artifacts.directory}")
        return 0

    raise RuntimeError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
