"""Command-line interface of the tool (thesis Section 7.1).

Subcommands:
  co      -- collectively optimal synthesis smt(F, f) (Definition 4.6.1,
             Theorem 4.6.1)
  ibis    -- Algorithm 1: IBIS
  sibis   -- Algorithm 2: S-IBIS (feasibility filter and binary search)
  qibis   -- Algorithm 3: Q-IBIS (Section 4.9)
  cegar   -- Algorithm 4: CEGAR-NE (Chapter 6)
  train   -- Q-learning self-play (Section 2.4), not used by the algorithms
  certify -- independent certifier of logged equilibria (Section 7.5)

The horizon defaults to ``m = floor(|T|/|Agt|)`` (Definition 3.9.1), the only
horizon IBIS, S-IBIS, Q-IBIS and CEGAR-NE accept, and the iteration bound
``N`` defaults to 30; ``--max-iters none`` runs without a bound, which stops
at an equilibrium or at the first recurring play.  ``--initial collective``
takes the initial play from ``smt(F, f)`` (Definition 4.6.1) instead of
``smt(F)``; for ``cegar`` it selects the initial query at the discrete
partition, and ``--inner-initial`` the one at a coarse partition (default:
``--initial``).  Plays are printed in the numbering of
the thesis: agents ``1..n`` and sectors ``1..|T|`` in the enumeration of
Remark 3.1.1 (bottom row first, left to right).  The tool numbers both from
0 internally.

Exit codes: 0 when ``co`` finds a play or a synthesiser returns an
equilibrium (outcome ``ne``), 2 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from types import EllipsisType

from .model import Coord, Territory
from .smt_solve import INITIAL_QUERIES, SmtSolution
from .strategy import Play

DEFAULT_MAX_ITERS = 30
"""The default iteration bound ``N`` (that of the first run of Chapter 7)."""

NO_BOUND = "none"
"""The value of ``--max-iters`` (and the CEGAR-NE bounds) for no bound."""


def parse_bound(text: str) -> int | None:
    """An iteration bound: a non-negative integer, or ``none`` for no bound."""
    if text.strip().lower() == NO_BOUND:
        return None
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer or '{NO_BOUND}', got {text!r}"
        ) from None
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"an iteration bound must be >= 0 or '{NO_BOUND}', got {value}"
        )
    return value


@dataclass(frozen=True, slots=True)
class RunArgs:
    mode: str
    grid_path: str
    agents: int
    horizon: int | None
    max_iters: int | None
    progress: bool
    timing: bool
    timeout_ms: int | None
    dump_model: bool
    render: bool
    symmetry: bool
    partition: str
    synthesiser: str
    demands: tuple[int, ...] | None
    seed: int = 0
    top_k: int = 3
    initial: str = "satisfying"
    inner_max_iters: int | None | EllipsisType = ...
    discrete_max_iters: int | None | EllipsisType = ...
    inner_initial: str | EllipsisType = ...


def _agent_symbol(o: int) -> str:
    """Thesis number of tool agent *o* as one character (``.`` if uncontrolled)."""
    if o < 0:
        return "."
    if o + 1 < 10:
        return str(o + 1)
    return chr(ord("A") + (o + 1 - 10) % 26)


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
            row.append(" " if idx is None else _agent_symbol(owner_by_index[idx]))
        lines.append("".join(row))
    return "\n".join(lines)


def _render_trace(sol: SmtSolution, territory: Territory) -> None:
    if sol.owner_by_round is None:
        return
    index_of = {c: i for i, c in enumerate(territory.ordered_sectors())}
    print("\n=== TRACE (t=0..m; agents 1..n, '.' uncontrolled) ===")
    for t, snapshot in enumerate(sol.owner_by_round):
        print(f"\n-- t={t} --")
        print(_render_owner_grid(territory=territory, owner_by_index=snapshot))
        if sol.actions_by_round is not None and t < len(sol.actions_by_round):
            step = sol.actions_by_round[t]
            action_str = " ".join(
                f"{a + 1}->{index_of[c] + 1 if c is not None else '-'}"
                for a, c in enumerate(step)
            )
            print(f"actions: {action_str}")


def _print_play(play: Play | None, num_agents: int) -> None:
    """Print the claims of every agent, in the numbering of the thesis."""
    if play is None:
        return
    print("play (agent: sectors claimed in rounds 0..m-1, thesis numbering):")
    for a in range(num_agents):
        claims = ", ".join(str(i + 1) for i in play.claims_of(a))
        print(f"  agent {a + 1}: {claims}")


def _parse_demands(raw: str, num_agents: int) -> tuple[int, ...]:
    """Parse a comma-separated demand string into a tuple of ints."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != num_agents:
        raise SystemExit(
            f"--demands requires exactly {num_agents} comma-separated integers, "
            f"got {len(parts)}"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--demands values must be integers, got: {raw}")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    p.add_argument("--grid", required=True, help="Path to an ASCII grid file")
    p.add_argument("--agents", type=int, required=True, help="Number of agents")
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Horizon m (default floor(|T|/|Agt|), the only value the "
        "synthesisers accept)",
    )
    p.add_argument(
        "--demands",
        type=str,
        default=None,
        help="Comma-separated per-agent demands d(a) (e.g. '3,3,5,5'); default 0",
    )


def _add_synthesis_args(p: argparse.ArgumentParser, *, max_iters_help: str) -> None:
    p.add_argument(
        "--max-iters",
        type=parse_bound,
        default=DEFAULT_MAX_ITERS,
        metavar="N",
        help=max_iters_help,
    )
    p.add_argument(
        "--initial",
        choices=list(INITIAL_QUERIES),
        default="satisfying",
        help="Query for the initial play: satisfying = smt(F) (default), "
        "collective = smt(F, f) of Definition 4.6.1",
    )
    p.add_argument(
        "--timeout-ms",
        type=int,
        default=0,
        help="Z3 timeout per query (0 = unset); a query that times out stops "
        "the run with outcome unk",
    )
    p.add_argument(
        "--progress", action="store_true", help="Print per-iteration progress"
    )
    p.add_argument("--timing", action="store_true", help="Include solve timings")
    p.add_argument("--render", action="store_true", help="Print ASCII state trace")
    p.add_argument(
        "--symmetry",
        action="store_true",
        help="Add the symmetry-breaking constraints of Chapter 5 to the "
        "initial query",
    )


_BOUND_HELP = (
    "Iteration bound N: the loop runs i = 0..N (default 30); 'none' runs "
    "without a bound until an equilibrium or a recurring play"
)


def _parse_args(argv: list[str] | None = None) -> RunArgs:
    p = argparse.ArgumentParser(prog="rcmas")
    sub = p.add_subparsers(dest="command", required=True)

    # -- co: collective optimality (Definition 4.6.1) ------------------
    co = sub.add_parser(
        "co", help="Collectively optimal play via smt(F, f) (Definition 4.6.1)"
    )
    _add_common_args(co)
    co.add_argument(
        "--timeout-ms", type=int, default=0, help="Z3 timeout per solve (0=unset)"
    )
    co.add_argument("--render", action="store_true", help="Print ASCII state trace")
    co.add_argument("--dump-model", action="store_true", help="Print model variables")
    co.add_argument(
        "--symmetry",
        action="store_true",
        help="Add the symmetry-breaking constraints of Chapter 5",
    )

    # -- ibis: Algorithm 1 ---------------------------------------------
    ibis = sub.add_parser("ibis", help="Nash equilibrium via IBIS (Algorithm 1)")
    _add_common_args(ibis)
    _add_synthesis_args(ibis, max_iters_help=_BOUND_HELP)

    # -- sibis: Algorithm 2 --------------------------------------------
    sibis = sub.add_parser(
        "sibis",
        help="Nash equilibrium via S-IBIS (Algorithm 2: feasibility filter and "
        "binary search)",
    )
    _add_common_args(sibis)
    _add_synthesis_args(sibis, max_iters_help=_BOUND_HELP)

    # -- qibis: Algorithm 3 --------------------------------------------
    qibis = sub.add_parser(
        "qibis", help="Nash equilibrium via Q-IBIS (Algorithm 3, Section 4.9)"
    )
    _add_common_args(qibis)
    _add_synthesis_args(qibis, max_iters_help=_BOUND_HELP)
    qibis.add_argument(
        "--seed", type=int, default=0, help="Seed of the Q-learners (default 0)"
    )
    qibis.add_argument(
        "--top-k", type=int, default=3, help="k of TopK (default 3, Section 4.9)"
    )

    # -- cegar: Algorithm 4 --------------------------------------------
    cegar = sub.add_parser(
        "cegar", help="Nash equilibrium via CEGAR-NE (Algorithm 4, Chapter 6)"
    )
    _add_common_args(cegar)
    cegar.add_argument(
        "--partition",
        default="orbit",
        help="Initial partition P_0: orbit (default, Section 6.5), discrete, "
        "trivial, balanced (n equal strips), grid-WxH (e.g. grid-2x2), bfs-K "
        "(e.g. bfs-4)",
    )
    cegar.add_argument(
        "--synthesiser",
        choices=["ibis", "qibis", "sibis"],
        default="sibis",
        help="Inner synthesiser SynthesiseNE (default sibis, as in Chapter 7)",
    )
    _add_synthesis_args(
        cegar,
        max_iters_help="Iteration bound N of the inner synthesiser at every "
        "partition (default 30; 'none' for no bound); Algorithm 4 itself "
        "needs no bound (Theorem 6.4.2)",
    )
    cegar.add_argument(
        "--inner-max-iters",
        type=parse_bound,
        default=...,
        metavar="N",
        help="Bound of the inner synthesiser at a coarse partition (default: "
        "--max-iters)",
    )
    cegar.add_argument(
        "--discrete-max-iters",
        type=parse_bound,
        default=...,
        metavar="N",
        help="Bound of the inner synthesiser at the discrete partition "
        "(default: --max-iters; 'none' as in the thesis protocol)",
    )
    cegar.add_argument(
        "--inner-initial",
        choices=list(INITIAL_QUERIES),
        default=...,
        help="Initial query of the inner synthesiser at a coarse partition, "
        "on the weighted abstract game (default: --initial, which then "
        "applies at the discrete partition only; satisfying as in the thesis "
        "protocol)",
    )
    cegar.add_argument(
        "--seed", type=int, default=0, help="Seed of an inner Q-IBIS (default 0)"
    )

    # -- train: Q-learning self-play (Section 2.4) ---------------------
    train = sub.add_parser("train", help="Q-learning self-play training (Section 2.4)")
    _add_common_args(train)

    # -- certify: independent certifier (Section 7.5) ------------------
    # Listed here for `rcmas --help`; main() hands `rcmas certify ...` to
    # rcmas.certify.main, which needs no grid or agent count.
    from .certify import add_arguments as _add_certify_args

    _add_certify_args(
        sub.add_parser(
            "certify",
            help="Re-check the equilibria of result logs without SMT (Section 7.5)",
        )
    )

    ns = p.parse_args(argv)
    if ns.agents <= 0:
        raise SystemExit("--agents must be >= 1")
    if ns.horizon is not None and ns.horizon < 0:
        raise SystemExit("--horizon must be >= 0")

    max_iters = getattr(ns, "max_iters", DEFAULT_MAX_ITERS)

    timeout_ms_raw = getattr(ns, "timeout_ms", 0)
    if timeout_ms_raw < 0:
        raise SystemExit("--timeout-ms must be >= 0")

    top_k = getattr(ns, "top_k", 3)
    if top_k < 0:
        raise SystemExit("--top-k must be >= 0")

    demands_raw = getattr(ns, "demands", None)
    demands = (
        _parse_demands(demands_raw, ns.agents) if demands_raw is not None else None
    )

    return RunArgs(
        mode=ns.command,
        grid_path=ns.grid,
        agents=ns.agents,
        horizon=ns.horizon,
        max_iters=max_iters,
        progress=bool(getattr(ns, "progress", False)),
        timing=bool(getattr(ns, "timing", False)),
        timeout_ms=None if int(timeout_ms_raw) == 0 else int(timeout_ms_raw),
        dump_model=bool(getattr(ns, "dump_model", False)),
        render=bool(getattr(ns, "render", False)),
        symmetry=bool(getattr(ns, "symmetry", False)),
        partition=str(getattr(ns, "partition", "orbit")),
        synthesiser=str(getattr(ns, "synthesiser", "sibis")),
        demands=demands,
        seed=int(getattr(ns, "seed", 0)),
        top_k=int(top_k),
        initial=str(getattr(ns, "initial", "satisfying")),
        inner_max_iters=getattr(ns, "inner_max_iters", ...),
        discrete_max_iters=getattr(ns, "discrete_max_iters", ...),
        inner_initial=getattr(ns, "inner_initial", ...),
    )


def _report(res: object, args: RunArgs, territory: Territory) -> int:
    """Print the result of a synthesiser and return the exit code."""
    outcome = getattr(res, "outcome", "")
    reason = getattr(res, "reason", "")
    print(
        f"outcome={outcome} reason={reason} found_ne={getattr(res, 'found_ne', False)}"
    )
    rewards = getattr(res, "payoff_by_agent", None)
    if rewards is not None:
        print(f"rewards={tuple(rewards)}")
    print(f"iterations={getattr(res, 'iterations', None)}")
    init_s = getattr(res, "initial_query_s", None)
    init_r = getattr(res, "initial_rewards", None)
    queries = f"initial={args.initial}"
    if args.mode == "cegar":
        # The time and rewards are those of the first inner call, at a coarse
        # partition unless P_0 is discrete.
        inner = args.initial if args.inner_initial is ... else args.inner_initial
        queries += f" inner_initial={inner}"
    print(
        f"{queries} initial_query_s="
        f"{'-' if init_s is None else f'{init_s:.3f}'} "
        f"initial_rewards={None if init_r is None else tuple(init_r)} "
        f"returned_initial={getattr(res, 'returned_initial', False)}"
    )
    _print_play(getattr(res, "play", None), args.agents)
    final = getattr(res, "final_solution", None)
    if args.render and final is not None:
        _render_trace(final, territory)
    return 0 if outcome == "ne" else 2


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    if raw_argv[:1] == ["certify"]:
        from .certify import main as certify_main

        return certify_main(raw_argv[1:])
    try:
        args = _parse_args(argv)
    except BrokenPipeError:  # pragma: no cover
        return 0

    with open(args.grid_path, "r", encoding="utf-8") as f:
        territory = Territory.from_ascii(f)

    m = len(territory) // args.agents
    horizon = m if args.horizon is None else args.horizon
    if args.mode in ("ibis", "sibis", "qibis", "cegar") and horizon != m:
        raise SystemExit(
            f"--horizon must be m = floor(|T|/|Agt|) = {m} for {args.mode} "
            f"(Definition 3.9.1), got {horizon}; omit --horizon to use m"
        )
    timeout_ms = args.timeout_ms

    if args.mode == "co":
        from .smt_solve import solve_collective_optimality

        debug = args.dump_model or args.render
        sol = solve_collective_optimality(
            territory=territory,
            num_agents=args.agents,
            horizon=horizon,
            debug=debug,
            symmetry_breaking=args.symmetry,
            demands=args.demands,
            timeout_ms=timeout_ms,
        )
        print(f"sat={sol.is_sat} reason={sol.reason} horizon={horizon}")
        if sol.payoff_by_agent is not None:
            print(f"rewards={sol.payoff_by_agent} welfare={sum(sol.payoff_by_agent)}")
        _print_play(sol.play, args.agents)

        if args.render:
            _render_trace(sol, territory)

        if args.dump_model and sol.final_state is not None:
            sectors = territory.ordered_sectors()
            owner = sol.final_state.owner_by_index
            for a in range(args.agents):
                owned = [sectors[i] for i, o in enumerate(owner) if o == a]
                print(f"agent[{a + 1}] controlled={len(owned)}")
                if sol.payoff_by_agent is not None:
                    print(f"agent[{a + 1}] reward={sol.payoff_by_agent[a]}")
                if sol.encoded_reward_by_agent is not None:
                    print(
                        f"agent[{a + 1}] encoded_reward={sol.encoded_reward_by_agent[a]}"
                    )
                if sol.best_seed_by_agent is not None:
                    print(f"agent[{a + 1}] witness_root={sol.best_seed_by_agent[a]}")
                region = (
                    sol.best_region_by_agent[a]
                    if sol.best_region_by_agent is not None
                    else None
                )
                if region is not None:
                    print(f"agent[{a + 1}] witness_region_size={len(region)}")
                    print(f"agent[{a + 1}] witness_region={region}")
            if sol.actions_by_round is not None:
                for t, step in enumerate(sol.actions_by_round):
                    print(f"t={t} actions={step}")

        return 0 if sol.is_sat else 2

    if args.mode in ("ibis", "sibis"):
        from .ibis import solve_ibis, solve_sibis

        solve = solve_ibis if args.mode == "ibis" else solve_sibis
        res = solve(
            territory=territory,
            num_agents=args.agents,
            horizon=horizon,
            max_iters=args.max_iters,
            initial=args.initial,
            progress=args.progress,
            timing=args.timing,
            timeout_ms=timeout_ms,
            symmetry=args.symmetry,
            demands=args.demands,
        )
        return _report(res, args, territory)

    if args.mode == "qibis":
        from .qibis import QibisConfig, solve_qibis

        cfg = QibisConfig(
            max_iters=args.max_iters,
            initial=args.initial,
            progress=args.progress,
            timing=args.timing,
            timeout_ms=timeout_ms,
            symmetry=args.symmetry,
            demands=args.demands,
            seed=args.seed,
            top_k=args.top_k,
        )
        qres = solve_qibis(
            territory=territory, num_agents=args.agents, horizon=horizon, cfg=cfg
        )
        return _report(qres, args, territory)

    if args.mode == "cegar":
        from .cegar import solve_cegar

        cres = solve_cegar(
            territory=territory,
            num_agents=args.agents,
            horizon=horizon,
            initial_partition=args.partition,
            synthesiser=args.synthesiser,
            max_iters=args.max_iters,
            inner_max_iters=args.inner_max_iters,
            discrete_max_iters=args.discrete_max_iters,
            initial=args.initial,
            inner_initial=args.inner_initial,
            progress=args.progress,
            timing=args.timing,
            timeout_ms=timeout_ms,
            symmetry=args.symmetry,
            demands=args.demands,
            seed=args.seed,
        )
        code = _report(cres, args, territory)
        print(
            f"initial_partition_size={cres.initial_partition_size} "
            f"final_partition_size={cres.final_partition_size} "
            f"at_discrete={cres.at_discrete} refinements={cres.refinements} "
            f"inner_iterations={cres.inner_iterations}"
        )
        return code

    if args.mode == "train":
        from .qlearning import TrainConfig, train_self_play

        tcfg = TrainConfig(episodes=2000, max_rounds=horizon)
        artifacts = train_self_play(territory, num_agents=args.agents, cfg=tcfg, seed=0)
        print(f"trained {len(artifacts.qtables)} agents")
        print(f"artifacts at {artifacts.directory}")
        return 0

    raise SystemExit(f"unknown command: {args.mode}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
