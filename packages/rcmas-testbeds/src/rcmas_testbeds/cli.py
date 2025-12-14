from __future__ import annotations

import argparse
from dataclasses import dataclass

from rcmas_core.engine import Coord, GameEngine, GameState, Territory

from .testbeds.registry import build_testbed


@dataclass(frozen=True, slots=True)
class RunArgs:
    testbed: str
    grid_path: str
    agents: int
    max_rounds: int
    max_iters: int
    progress: bool
    timing: bool
    timeout_ms: int | None
    dump_model: bool
    render: bool


def _render_owner_grid(*, territory: Territory, owner_by_index: tuple[int, ...]) -> str:
    sectors = territory.ordered_sectors()
    if not sectors:
        return "(empty territory)"

    sector_index_by_coord = {c: i for i, c in enumerate(sectors)}

    max_x = max(c.x for c in sectors)
    max_y = max(c.y for c in sectors)

    lines: list[str] = []
    for y in range(max_y + 1):
        row: list[str] = []
        for x in range(max_x + 1):
            idx = sector_index_by_coord.get(Coord(x=x, y=y))
            if idx is None:
                row.append(" ")
                continue

            o = owner_by_index[idx]
            if o < 0:
                row.append(".")
            elif o < 10:
                row.append(str(o))
            else:
                row.append(chr(ord("A") + (o - 10) % 26))
        lines.append("".join(row))
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> RunArgs:
    p = argparse.ArgumentParser(prog="rcmas-testbeds")
    p.add_argument(
        "--testbed",
        required=True,
        choices=["smt-co", "qlearning", "smt-ne", "hybrid"],
        help="Which experiment mode to run",
    )
    p.add_argument("--grid", required=True, help="Path to an ASCII grid file")
    p.add_argument("--agents", type=int, required=True, help="Number of agents")
    p.add_argument("--max-rounds", type=int, default=10, help="Max rounds")
    p.add_argument("--max-iters", type=int, default=25, help="For smt-ne: max best-response iterations")
    p.add_argument("--progress", action="store_true", help="For smt-ne: print per-iteration progress to stderr")
    p.add_argument("--timing", action="store_true", help="For smt-ne: include solve timings in progress output")
    p.add_argument("--timeout-ms", type=int, default=0, help="For smt-co/smt-ne: Z3 timeout per solve (0=unset)")
    p.add_argument("--dump-model", action="store_true", help="For smt-co: print model variables")
    p.add_argument("--render", action="store_true", help="For smt-co: print an ASCII state trace")

    ns = p.parse_args(argv)
    if ns.agents <= 0:
        raise SystemExit("--agents must be >= 1")
    if ns.max_rounds <= 0:
        raise SystemExit("--max-rounds must be >= 1")
    if ns.max_iters <= 0:
        raise SystemExit("--max-iters must be >= 1")
    if ns.timeout_ms < 0:
        raise SystemExit("--timeout-ms must be >= 0")

    return RunArgs(
        testbed=ns.testbed,
        grid_path=ns.grid,
        agents=ns.agents,
        max_rounds=ns.max_rounds,
        max_iters=int(ns.max_iters),
        progress=bool(ns.progress),
        timing=bool(ns.timing),
        timeout_ms=None if int(ns.timeout_ms) == 0 else int(ns.timeout_ms),
        dump_model=bool(ns.dump_model),
        render=bool(ns.render),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except BrokenPipeError:  # pragma: no cover
        return 0

    with open(args.grid_path, "r", encoding="utf-8") as f:
        territory = Territory.from_ascii(f)

    tb = build_testbed(
        args.testbed,
        max_iters=args.max_iters,
        progress=args.progress,
        timing=args.timing,
        timeout_ms=args.timeout_ms,
    )

    if args.testbed in {"smt-co", "smt-ne"}:
        if args.testbed == "smt-ne":
            res = tb.solve(territory=territory, num_agents=args.agents, horizon=args.max_rounds)  # type: ignore[attr-defined]
            print(f"sat={res.is_sat} reason={res.reason}")
            if res.payoff_by_agent is not None:
                print(f"payoff={res.payoff_by_agent}")
            print(f"iterations={res.iterations} found_ne={res.found_ne}")

            sol = res.final_solution
            if args.render and sol is not None and sol.owner_by_round is not None:
                print("\n=== TRACE (t=0..T) ===")
                for t, snapshot in enumerate(sol.owner_by_round):
                    print(f"\n-- t={t} --")
                    print(_render_owner_grid(territory=territory, owner_by_index=snapshot))
                    if sol.actions_by_round is not None and t < len(sol.actions_by_round):
                        step = sol.actions_by_round[t]
                        action_str = " ".join(f"a{a}={c}" for a, c in enumerate(step))
                        print(f"actions: {action_str}")

            return 0 if res.is_sat else 2

        # One-shot SMT: no interaction with the engine loop.
        if args.dump_model or args.render:
            sol = tb.solve_debug(territory=territory, num_agents=args.agents, horizon=args.max_rounds)  # type: ignore[attr-defined]
        else:
            sol = tb.solve(territory=territory, num_agents=args.agents, horizon=args.max_rounds)  # type: ignore[attr-defined]
        print(f"sat={sol.is_sat} reason={sol.reason}")
        if sol.final_state is not None:
            print(f"scores={sol.final_state.scores()}")

        if args.render and sol.owner_by_round is not None:
            print("\n=== TRACE (t=0..T) ===")
            for t, snapshot in enumerate(sol.owner_by_round):
                print(f"\n-- t={t} --")
                print(_render_owner_grid(territory=territory, owner_by_index=snapshot))
                if sol.actions_by_round is not None and t < len(sol.actions_by_round):
                    step = sol.actions_by_round[t]
                    action_str = " ".join(f"a{a}={c}" for a, c in enumerate(step))
                    print(f"actions: {action_str}")

        if args.dump_model and sol.final_state is not None:
            sectors = territory.ordered_sectors()
            owner = sol.final_state.owner_by_index
            for a in range(args.agents):
                owned = [sectors[i] for i, o in enumerate(owner) if o == a]
                print(f"agent[{a}] owned={len(owned)}")
                if sol.payoff_by_agent is not None:
                    print(f"agent[{a}] payoff={sol.payoff_by_agent[a]}")
                if sol.best_seed_by_agent is not None:
                    print(f"agent[{a}] best_seed={sol.best_seed_by_agent[a]}")
                if sol.best_region_by_agent is not None and sol.best_region_by_agent[a] is not None:
                    region = sol.best_region_by_agent[a]
                    print(f"agent[{a}] cohesive_region_size={len(region)}")
                    print(f"agent[{a}] cohesive_region={region}")

            if sol.actions_by_round is not None:
                for t, step in enumerate(sol.actions_by_round):
                    print(f"t={t} actions={step}")
        return 0 if sol.is_sat else 2

    state = GameState.new(territory, num_agents=args.agents)
    engine = GameEngine(state)
    result = engine.run(
        tb.build_agents(territory=territory, num_agents=args.agents, max_rounds=args.max_rounds),
        max_rounds=args.max_rounds,
    )

    print(f"outcome={result.outcome.status.value} reason={result.outcome.reason}")
    print(f"scores={result.scores}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
