#!/usr/bin/env python3
"""Command line of the experiment framework.

The thesis sweep (Chapter 7) is::

    python -m experiments.run --config experiments/thesis-eval.toml \\
        --output results/thesis-eval-900.jsonl --workers 0 --resume

(``--workers 0``: physical cores minus one, on Apple silicon performance
cores minus one.)

The ``[protocol]`` table of the configuration fixes the limit per run
(900 s), the number of runs (1), the iteration bound of the direct runs
(none), the initial query (collective), the bounds of the inner synthesiser
of CEGAR-NE (30 at a coarse partition, none at the discrete one), the demand
(0) and the Q-IBIS seed (0); ``experiments/thesis-eval-n30.toml`` holds the
protocol of the first run (N = 30 everywhere, satisfying initial query).
``--timeout``, ``--runs``, ``--max-iters``, ``--initial``,
``--cegar-inner-max-iters``, ``--cegar-discrete-max-iters``, ``--demand``
and ``--seed`` override it; a bound is an integer or ``none``.
``--max-iters`` also sets a CEGAR-NE bound that neither the configuration
nor the command line sets.  The values used are recorded in the session
header of the log and in every run record.  ``./reproduce.sh`` runs the
whole pipeline.

Other uses::

    python -m experiments.run --config experiments/thesis-eval.toml --list
    python -m experiments.run --config experiments/thesis-eval.toml \\
        --output results/thesis-eval-900.jsonl --check-resume
    python -m experiments.run --analyze results/thesis-eval-900.jsonl
    python -m experiments.run --suites E1 --runs 1 --timeout 60 --output e1.jsonl

``--check-resume`` runs nothing: it says whether the sweep could resume the
log given by ``--output`` and exits with 0 (no log, or a compatible one), 3
(the log has another synthesis protocol, for example that of the first run)
or 4 (another commit, Z3 version, configuration or protocol field).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .config import (
    INITIAL_QUERIES,
    Protocol,
    load_config,
    load_protocol,
    parse_bound,
    protocol_table,
)
from .provenance import default_workers
from .runner import (
    LogInUseError,
    MixedLogError,
    SessionTerminated,
    check_resume,
    describe_protocol,
    run_batch,
)

CEGAR_BOUNDS = ("cegar_inner_max_iters", "cegar_discrete_max_iters")


def _bound_arg(text: str) -> int | None:
    try:
        return parse_bound(text, "the bound")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _summary(records: list[dict]) -> None:
    from collections import Counter, defaultdict

    by_cfg: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        cfg = r["algorithm"] + ("+sym" if r["symmetry"] else "")
        by_cfg[cfg][r.get("outcome") or "?"] += 1
        if r.get("solved"):
            by_cfg[cfg]["solved"] += 1
    print(
        "\nOutcomes per configuration (solved = equilibrium accepted by the certifier):",
        file=sys.stderr,
    )
    for cfg in sorted(by_cfg):
        c = by_cfg[cfg]
        parts = ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if k != "solved")
        print(f"  {cfg:18s} solved {c['solved']:3d}  ({parts})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experiments",
        description="RCMAS experiment framework (thesis Chapter 7)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="TOML_FILE",
        help="Load scenarios and the protocol from a TOML config file (overrides --suites)",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default=None,
        help="Legacy symmetry suites (E1..E6) instead of --config",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Runs per scenario (default: config, else 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Wall-clock limit per run in seconds (default: config, else 900)",
    )
    parser.add_argument(
        "--max-iters",
        type=_bound_arg,
        default=...,
        metavar="N",
        help="Iteration bound N of IBIS, S-IBIS and Q-IBIS, an integer or 'none' "
        "(default: config, else 30); also sets a CEGAR-NE bound that is not set "
        "otherwise",
    )
    parser.add_argument(
        "--initial",
        choices=list(INITIAL_QUERIES),
        default=None,
        help="Query for the initial play (default: config, else satisfying)",
    )
    parser.add_argument(
        "--cegar-inner-max-iters",
        type=_bound_arg,
        default=...,
        metavar="N",
        help="Bound of the inner synthesiser of CEGAR-NE at a coarse partition "
        "(default: config, else the bound of the direct runs)",
    )
    parser.add_argument(
        "--cegar-discrete-max-iters",
        type=_bound_arg,
        default=...,
        metavar="N",
        help="Bound of the inner synthesiser of CEGAR-NE at the discrete "
        "partition (default: config, else the bound of the direct runs)",
    )
    parser.add_argument(
        "--demand",
        type=int,
        default=None,
        help="Demand of every agent (default: config, else 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed of the Q-IBIS learners (default: config, else 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Append JSON-lines records to this file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing output file, skipping the runs it already holds",
    )
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help="Resume even if the log was written by another tool commit, other uncommitted "
        "changes, another Z3 version, configuration or protocol",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Runs in parallel, one process each (default 1; 0 = physical cores minus one)",
    )
    parser.add_argument(
        "--no-certify",
        action="store_true",
        help="Do not certify returned equilibria during the sweep",
    )
    parser.add_argument(
        "--cert-timeout",
        type=float,
        default=1800.0,
        help="Time limit of the certifier per equilibrium in seconds (default 1800)",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        default=None,
        metavar="RESULTS_FILE",
        help="Analyse a results file instead of running",
    )
    parser.add_argument(
        "--random-grids",
        type=int,
        default=20,
        help="Random grids for suite E6 (default 20)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Seed for random grids (default 42)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List scenarios without running them"
    )
    parser.add_argument(
        "--check-resume",
        action="store_true",
        help="Run nothing; say whether the sweep could resume --output (exit 0: "
        "yes or no log, 3: another synthesis protocol, 4: another setup)",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Suppress per-run progress output"
    )

    args = parser.parse_args(argv)

    if args.analyze:
        from .analyze import full_report, load_results

        results = load_results(args.analyze)
        print(f"Loaded {len(results)} results from {args.analyze}", file=sys.stderr)
        full_report(results)
        return 0

    if args.config:
        scenarios = load_config(args.config)
        protocol = load_protocol(args.config)
    else:
        from .gridgen import e6_random_scenarios
        from .scenarios import all_scenarios

        suite_list = args.suites.split(",") if args.suites else None
        scenarios = all_scenarios(suites=suite_list)
        if suite_list is None or "E6" in suite_list:
            scenarios.extend(
                e6_random_scenarios(num_grids=args.random_grids, seed=args.random_seed)
            )
        protocol = Protocol()

    overrides: dict[str, object] = {
        k: v
        for k, v in {
            "runs": args.runs,
            "time_limit_s": args.timeout,
            "demand": args.demand,
            "seed": args.seed,
            "initial": args.initial,
        }.items()
        if v is not None
    }
    table = protocol_table(args.config) if args.config else {}
    for key in CEGAR_BOUNDS:
        value = getattr(args, key)
        if value is not ...:
            overrides[key] = value
        elif args.max_iters is not ... and key not in table:
            overrides[key] = args.max_iters
    if args.max_iters is not ...:
        overrides["max_iters"] = args.max_iters
    protocol = replace(protocol, **overrides)  # type: ignore[arg-type]

    if args.check_resume:
        if not args.output:
            parser.error("--check-resume needs --output")
        check = check_resume(args.output, protocol=protocol, config_path=args.config)
        print(f"{check.status}: {args.output} ({describe_protocol(protocol)})")
        for line in check.problems:
            print(f"  {line}")
        return {"new": 0, "ok": 0, "protocol": 3}.get(check.status, 4)

    if not scenarios:
        print("No scenarios matched.", file=sys.stderr)
        return 1

    if args.list:
        from rcmas.model import Territory

        print(
            f"{'Suite':>11s} {'Algorithm':>12s} {'Partition':>10s} {'Sym':>5s} {'S':>3s} {'n':>2s} {'m':>3s} Name"
        )
        instances = set()
        for s in scenarios:
            with open(s.grid_path, encoding="utf-8") as f:
                S = len(Territory.from_ascii(f))
            instances.add((s.grid_path, s.num_agents))
            part = s.partition if s.algorithm.startswith("cegar") else "-"
            print(
                f"{s.suite:>11s} {s.algorithm:>12s} {part:>10s} {str(s.symmetry):>5s} {S:>3d} {s.num_agents:>2d} {s.horizon:>3d} {s.name}"
            )
        print(
            f"\nTotal: {len(instances)} instances, {len(scenarios)} scenarios x {protocol.runs} runs "
            f"= {len(scenarios) * protocol.runs} runs; protocol {protocol.as_dict()}"
        )
        return 0

    workers = args.workers if args.workers > 0 else default_workers()
    try:
        records = run_batch(
            scenarios,
            protocol=protocol,
            output_path=args.output,
            workers=workers,
            resume=args.resume,
            progress=not args.no_progress,
            certify=not args.no_certify,
            cert_timeout_s=args.cert_timeout,
            config_path=args.config,
            argv=sys.argv if argv is None else ["python -m experiments.run", *argv],
            allow_mixed=args.allow_mixed,
        )
    except (MixedLogError, LogInUseError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt as e:
        how = e.signame if isinstance(e, SessionTerminated) else "Ctrl-C"
        print(
            f"\nStopped by {how}; the running runs were killed and the finished ones "
            "are in the log. Rerun with --resume to continue.",
            file=sys.stderr,
        )
        return 128 + e.signum if isinstance(e, SessionTerminated) else 130

    if not args.no_progress:
        _summary(records)
    if args.output:
        print(f"\nResults written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
