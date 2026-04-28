#!/usr/bin/env python3
"""CLI entry point for the experiment framework.

Usage:
    # Run all experiment suites
    python -m experiments.run --output results.jsonl

    # Run specific suites
    python -m experiments.run --suites E1,E3 --output results.jsonl

    # Run with random scenarios
    python -m experiments.run --suites E6 --random-grids 20 --output results.jsonl

    # Analyze existing results
    python -m experiments.run --analyze results.jsonl

    # Quick smoke test (1 run, 60s timeout, small suites)
    python -m experiments.run --suites E1 --runs 1 --timeout 60 --output smoke.jsonl

    # List scenarios without running
    python -m experiments.run --suites E1 --list
"""

from __future__ import annotations

import argparse
import sys

from .analyze import full_report, load_results
from .config import load_config
from .gridgen import e6_random_scenarios
from .runner import run_batch
from .scenarios import all_scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experiments",
        description="RCMAS experiment framework",
    )
    parser.add_argument(
        "--suites", type=str, default=None,
        help="Comma-separated suite names to run (E1,E2,E3,E4,E5,E6,E7). Default: all.",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of repeated runs for median aggregation (default: 3)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-scenario wall-clock timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write JSON-lines results to this file",
    )
    parser.add_argument(
        "--analyze", type=str, default=None, metavar="RESULTS_FILE",
        help="Analyze existing results file instead of running experiments",
    )
    parser.add_argument(
        "--random-grids", type=int, default=20,
        help="Number of random grids to generate for E6 (default: 20)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Seed for random grid generation (default: 42)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List scenarios without running them",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Suppress per-scenario progress output",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (default: 1 = sequential, 0 = all cores)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file, skipping completed experiments",
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="TOML_FILE",
        help="Load scenarios from a TOML config file (overrides --suites)",
    )

    args = parser.parse_args(argv)

    # Analysis mode
    if args.analyze:
        results = load_results(args.analyze)
        print(f"Loaded {len(results)} results from {args.analyze}", file=sys.stderr)
        full_report(results)
        return 0

    # Build scenario list
    if args.config:
        scenarios = load_config(args.config)
    else:
        suite_list = args.suites.split(",") if args.suites else None
        scenarios = all_scenarios(suites=suite_list)

        # Add E6 random scenarios if requested
        if suite_list is None or "E6" in suite_list:
            scenarios.extend(e6_random_scenarios(
                num_grids=args.random_grids,
                seed=args.random_seed,
            ))

    if not scenarios:
        print("No scenarios matched.", file=sys.stderr)
        return 1

    # List mode
    if args.list:
        print(f"{'Suite':>5s} {'Algorithm':>12s} {'Partition':>10s} {'Sym':>5s} {'S':>3s} {'n':>2s} {'h':>3s} {'Name'}")
        seen = set()
        for s in scenarios:
            key = (s.suite, s.algorithm, s.symmetry, s.partition, s.name)
            if key in seen:
                continue
            seen.add(key)
            with open(s.grid_path) as f:
                from rcmas.model import Territory
                t = Territory.from_ascii(f)
            S = len(t.sectors)
            part = s.partition if s.algorithm.startswith("cegar") else "-"
            print(f"{s.suite:>5s} {s.algorithm:>12s} {part:>10s} {str(s.symmetry):>5s} {S:>3d} {s.num_agents:>2d} {s.horizon:>3d} {s.name}")
        print(f"\nTotal: {len(seen)} unique scenarios x {args.runs} runs = {len(seen) * args.runs} executions")
        return 0

    # Run experiments
    print(f"Running {len(scenarios)} scenarios x {args.runs} runs = {len(scenarios) * args.runs} total", file=sys.stderr)
    results = run_batch(
        scenarios,
        num_runs=args.runs,
        timeout_s=args.timeout,
        output_path=args.output,
        progress=not args.no_progress,
        workers=args.workers,
        resume=args.resume,
    )

    # Print analysis
    from .analyze import ResultRow
    result_rows = [
        ResultRow(**{
            "suite": r.suite, "name": r.name, "grid_path": r.grid_path,
            "num_sectors": r.num_sectors, "num_agents": r.num_agents,
            "horizon": r.horizon, "algorithm": r.algorithm, "symmetry": r.symmetry,
            "partition": r.partition, "run": r.run, "status": r.status,
            "found_ne": r.found_ne, "payoff": r.payoff, "iterations": r.iterations,
            "time_s": r.time_s, "reason": r.reason,
            "cegar_final_blocks": r.cegar_final_blocks,
        })
        for r in results
    ]
    full_report(result_rows)

    if args.output:
        print(f"\nResults written to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
