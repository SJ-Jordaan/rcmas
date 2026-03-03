"""CLI for RCMAS.

Subcommands map to paper constructs:
  co    — Collective Optimality (Def 24)
  ibis  — Algorithm 1: IBIS
  qibis — Algorithm 2: Q-IBIS
  cegar — Algorithm 1 (EUMAS): CEGAR-NE
  train — Q-learning self-play (Sec 5.2)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .model import Coord, State, Territory, scores
from .smt_solve import SmtSolution


@dataclass(frozen=True, slots=True)
class RunArgs:
    mode: str
    grid_path: str
    agents: int
    horizon: int
    max_iters: int
    progress: bool
    timing: bool
    timeout_ms: int | None
    dump_model: bool
    render: bool
    symmetry: bool
    partition: str
    demands: tuple[int, ...] | None


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


def _render_trace(sol: SmtSolution, territory: Territory) -> None:
    if sol.owner_by_round is None:
        return
    print("\n=== TRACE (t=0..T) ===")
    for t, snapshot in enumerate(sol.owner_by_round):
        print(f"\n-- t={t} --")
        print(_render_owner_grid(territory=territory, owner_by_index=snapshot))
        if sol.actions_by_round is not None and t < len(sol.actions_by_round):
            step = sol.actions_by_round[t]
            action_str = " ".join(f"a{a}={c}" for a, c in enumerate(step))
            print(f"actions: {action_str}")


def _parse_demands(raw: str, num_agents: int) -> tuple[int, ...]:
    """Parse a comma-separated demand string into a tuple of ints."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != num_agents:
        raise SystemExit(f"--demands requires exactly {num_agents} comma-separated integers, got {len(parts)}")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--demands values must be integers, got: {raw}")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    p.add_argument("--grid", required=True, help="Path to an ASCII grid file")
    p.add_argument("--agents", type=int, required=True, help="Number of agents")
    p.add_argument("--horizon", type=int, default=10, help="Time horizon (max rounds)")
    p.add_argument("--demands", type=str, default=None, help="Comma-separated per-agent demands (e.g. '3,3,5,5')")


def _parse_args(argv: list[str] | None = None) -> RunArgs:
    p = argparse.ArgumentParser(prog="rcmas")
    sub = p.add_subparsers(dest="command", required=True)

    # ── co: Collective Optimality (Def 24) ────────────────────────
    co = sub.add_parser("co", help="Collective optimality via SMT (Def 24)")
    _add_common_args(co)
    co.add_argument("--timeout-ms", type=int, default=0, help="Z3 timeout per solve (0=unset)")
    co.add_argument("--render", action="store_true", help="Print ASCII state trace")
    co.add_argument("--dump-model", action="store_true", help="Print model variables")
    co.add_argument("--symmetry", action="store_true", help="Enable symmetry-breaking constraints")

    # ── ibis: Algorithm 1 (IBIS) ──────────────────────────────────
    ibis = sub.add_parser("ibis", help="Nash equilibrium via IBIS (Algorithm 1)")
    _add_common_args(ibis)
    ibis.add_argument("--max-iters", type=int, default=25, help="Max best-response iterations")
    ibis.add_argument("--timeout-ms", type=int, default=0, help="Z3 timeout per solve (0=unset)")
    ibis.add_argument("--progress", action="store_true", help="Print per-iteration progress")
    ibis.add_argument("--timing", action="store_true", help="Include solve timings")
    ibis.add_argument("--render", action="store_true", help="Print ASCII state trace")
    ibis.add_argument("--symmetry", action="store_true", help="Enable symmetry-breaking constraints")

    # ── qibis: Algorithm 2 (Q-IBIS) ──────────────────────────────
    qibis = sub.add_parser("qibis", help="Q-learning-guided IBIS (Algorithm 2)")
    _add_common_args(qibis)
    qibis.add_argument("--max-iters", type=int, default=25, help="Max best-response iterations")
    qibis.add_argument("--timeout-ms", type=int, default=0, help="Z3 timeout per solve (0=unset)")
    qibis.add_argument("--progress", action="store_true", help="Print per-iteration progress")
    qibis.add_argument("--timing", action="store_true", help="Include solve timings")
    qibis.add_argument("--render", action="store_true", help="Print ASCII state trace")
    qibis.add_argument("--symmetry", action="store_true", help="Enable symmetry-breaking constraints")

    # ── cegar: CEGAR-NE (EUMAS Algorithm 1) ─────────────────────
    cegar = sub.add_parser("cegar", help="CEGAR-NE abstraction refinement (EUMAS Algorithm 1)")
    _add_common_args(cegar)
    cegar.add_argument("--partition", choices=["orbit", "discrete"], default="orbit", help="Initial partition type")
    cegar.add_argument("--max-iters", type=int, default=25, help="Max refinement iterations")
    cegar.add_argument("--timeout-ms", type=int, default=0, help="Z3 timeout per solve (0=unset)")
    cegar.add_argument("--progress", action="store_true", help="Print per-iteration progress")
    cegar.add_argument("--timing", action="store_true", help="Include solve timings")
    cegar.add_argument("--render", action="store_true", help="Print ASCII state trace")
    cegar.add_argument("--symmetry", action="store_true", help="Enable symmetry-breaking constraints")

    # ── train: Q-learning self-play (Sec 5.2) ────────────────────
    train = sub.add_parser("train", help="Q-learning self-play training (Sec 5.2)")
    _add_common_args(train)

    ns = p.parse_args(argv)
    if ns.agents <= 0:
        raise SystemExit("--agents must be >= 1")
    if ns.horizon <= 0:
        raise SystemExit("--horizon must be >= 1")

    max_iters = getattr(ns, "max_iters", 25)
    if max_iters <= 0:
        raise SystemExit("--max-iters must be >= 1")

    timeout_ms_raw = getattr(ns, "timeout_ms", 0)
    if timeout_ms_raw < 0:
        raise SystemExit("--timeout-ms must be >= 0")

    demands_raw = getattr(ns, "demands", None)
    demands = _parse_demands(demands_raw, ns.agents) if demands_raw is not None else None

    return RunArgs(
        mode=ns.command,
        grid_path=ns.grid,
        agents=ns.agents,
        horizon=ns.horizon,
        max_iters=int(max_iters),
        progress=bool(getattr(ns, "progress", False)),
        timing=bool(getattr(ns, "timing", False)),
        timeout_ms=None if int(timeout_ms_raw) == 0 else int(timeout_ms_raw),
        dump_model=bool(getattr(ns, "dump_model", False)),
        render=bool(getattr(ns, "render", False)),
        symmetry=bool(getattr(ns, "symmetry", False)),
        partition=str(getattr(ns, "partition", "orbit")),
        demands=demands,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except BrokenPipeError:  # pragma: no cover
        return 0

    with open(args.grid_path, "r", encoding="utf-8") as f:
        territory = Territory.from_ascii(f)

    timeout_ms = args.timeout_ms

    if args.mode == "co":
        from .smt_solve import solve_collective_optimality

        debug = args.dump_model or args.render
        sol = solve_collective_optimality(
            territory=territory, num_agents=args.agents, horizon=args.horizon, debug=debug,
            symmetry_breaking=args.symmetry, demands=args.demands,
        )
        print(f"sat={sol.is_sat} reason={sol.reason}")
        if sol.final_state is not None:
            print(f"scores={scores(sol.final_state)}")

        if args.render:
            _render_trace(sol, territory)

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

    if args.mode == "ibis":
        from .ibis import solve_ibis

        res = solve_ibis(
            territory=territory, num_agents=args.agents, horizon=args.horizon,
            max_iters=args.max_iters, progress=args.progress,
            timing=args.timing, timeout_ms=timeout_ms,
            symmetry=args.symmetry, demands=args.demands,
        )
        print(f"sat={res.is_sat} reason={res.reason}")
        if res.payoff_by_agent is not None:
            print(f"payoff={res.payoff_by_agent}")
        print(f"iterations={res.iterations} found_ne={res.found_ne}")

        if args.render and res.final_solution is not None:
            _render_trace(res.final_solution, territory)

        return 0 if res.is_sat else 2

    if args.mode == "qibis":
        from .qibis import QibisConfig, solve_qibis

        cfg = QibisConfig(
            max_iters=args.max_iters, progress=args.progress,
            timing=args.timing, timeout_ms=timeout_ms,
            symmetry=args.symmetry, demands=args.demands,
        )
        res = solve_qibis(territory=territory, num_agents=args.agents, horizon=args.horizon, cfg=cfg)
        print(f"sat={res.is_sat} reason={res.reason}")
        if res.payoff_by_agent is not None:
            print(f"payoff={res.payoff_by_agent}")
        print(f"iterations={res.iterations} found_ne={res.found_ne}")

        if args.render and res.final_solution is not None:
            _render_trace(res.final_solution, territory)

        return 0 if res.is_sat else 2

    if args.mode == "cegar":
        from .cegar import solve_cegar

        res = solve_cegar(
            territory=territory, num_agents=args.agents, horizon=args.horizon,
            initial_partition=args.partition,
            max_iters=args.max_iters, progress=args.progress,
            timing=args.timing, timeout_ms=timeout_ms,
            symmetry=args.symmetry, demands=args.demands,
        )
        print(f"sat={res.is_sat} reason={res.reason}")
        if res.payoff_by_agent is not None:
            print(f"payoff={res.payoff_by_agent}")
        print(f"iterations={res.iterations} found_ne={res.found_ne} final_partition_size={res.final_partition_size}")

        if args.render and res.final_solution is not None:
            _render_trace(res.final_solution, territory)

        return 0 if res.is_sat else 2

    if args.mode == "train":
        from .qlearning import TrainConfig, train_self_play

        cfg = TrainConfig(episodes=2000, max_rounds=args.horizon)
        artifacts = train_self_play(territory, num_agents=args.agents, cfg=cfg, seed=0)
        print(f"trained {len(artifacts.qtables)} agents")
        print(f"artifacts at {artifacts.directory}")
        return 0

    raise SystemExit(f"unknown command: {args.mode}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
