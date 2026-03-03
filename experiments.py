#!/usr/bin/env python3
"""Benchmark IBIS vs CEGAR-NE across grid sizes, agent counts, and symmetry.

Usage:
    python experiments.py [--runs N] [--timeout SECS] [--filter PATTERN] [--output FILE] [--include-qibis]

Auto-discovers all grid files in grids/symmetric/ and grids/asymmetric/.
For each grid, generates all feasible (n, h) configs where S%n==0, n>=2, h>=2.
Outputs one JSON line per (config, algorithm, run) to stdout (or --output file).
Prints a summary table to stderr when complete.
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from collections import defaultdict
from statistics import median

from rcmas.cegar import solve_cegar
from rcmas.ibis import solve_ibis
from rcmas.model import Territory

# ---------------------------------------------------------------------------
# Grid directories
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYMMETRIC_DIR = os.path.join(SCRIPT_DIR, "grids", "symmetric")
ASYMMETRIC_DIR = os.path.join(SCRIPT_DIR, "grids", "asymmetric")


# ---------------------------------------------------------------------------
# Auto-discovery: scan grid dirs and compute all feasible configs
# ---------------------------------------------------------------------------

def discover_configs() -> list[tuple[str, str, str, int]]:
    """Scan grid directories and return (name, grid_type, path, S) for each grid."""
    grids = []
    for grid_dir, grid_type in [(SYMMETRIC_DIR, "symmetric"), (ASYMMETRIC_DIR, "asymmetric")]:
        if not os.path.isdir(grid_dir):
            continue
        for fname in sorted(os.listdir(grid_dir)):
            if not fname.endswith(".txt"):
                continue
            name = fname.replace(".txt", "")
            path = os.path.join(grid_dir, fname)
            with open(path) as f:
                territory = Territory.from_ascii(f)
            S = len(territory.sectors)
            grids.append((name, grid_type, path, S))
    return grids


def feasible_agent_counts(S: int) -> list[int]:
    """Return all n where S%n==0, n>=2, and h=S//n >= 2."""
    return [n for n in range(2, S + 1) if S % n == 0 and S // n >= 2]


# ---------------------------------------------------------------------------
# Timeout machinery (Unix signal-based)
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("solver timeout")


# ---------------------------------------------------------------------------
# Grid loading
# ---------------------------------------------------------------------------

def load_grid(path: str) -> Territory:
    with open(path) as f:
        return Territory.from_ascii(f)


# ---------------------------------------------------------------------------
# Algorithm runners — each returns a standardised result dict
# ---------------------------------------------------------------------------

def _run_ibis(
    territory: Territory, n: int, h: int, sym: bool, timeout_ms: int | None,
) -> dict:
    result = solve_ibis(
        territory=territory,
        num_agents=n,
        horizon=h,
        max_iters=30,
        symmetry=sym,
        timeout_ms=timeout_ms,
    )
    return {
        "ne": result.found_ne,
        "payoff": list(result.payoff_by_agent) if result.payoff_by_agent else None,
        "iters": result.iterations,
        "reason": result.reason,
        "cegar_final_blocks": None,
    }


def _run_cegar(
    territory: Territory, n: int, h: int, sym: bool, timeout_ms: int | None,
) -> dict:
    result = solve_cegar(
        territory=territory,
        num_agents=n,
        horizon=h,
        max_iters=25,
        symmetry=sym,
        timeout_ms=timeout_ms,
    )
    return {
        "ne": result.found_ne,
        "payoff": list(result.payoff_by_agent) if result.payoff_by_agent else None,
        "iters": result.iterations,
        "reason": result.reason,
        "cegar_final_blocks": result.final_partition_size,
    }


def _run_qibis(
    territory: Territory, n: int, h: int, sym: bool, timeout_ms: int | None,
) -> dict:
    from rcmas.qibis import QibisConfig, solve_qibis

    cfg = QibisConfig(max_iters=30, symmetry=sym, timeout_ms=timeout_ms)
    result = solve_qibis(territory=territory, num_agents=n, horizon=h, cfg=cfg)
    return {
        "ne": result.found_ne,
        "payoff": list(result.payoff_by_agent) if result.payoff_by_agent else None,
        "iters": result.iterations,
        "reason": result.reason,
        "cegar_final_blocks": None,
    }


# ---------------------------------------------------------------------------
# Single benchmark run with timeout
# ---------------------------------------------------------------------------

def run_single(
    territory: Territory,
    grid_name: str,
    grid_type: str,
    S: int,
    n: int,
    h: int,
    alg: str,
    sym: bool,
    timeout_s: int,
    run_num: int,
) -> dict:
    """Run a single (config, algorithm) benchmark and return a JSON-serialisable dict."""

    timeout_ms = timeout_s * 1000

    row_base = {
        "grid": grid_name,
        "grid_type": grid_type,
        "S": S,
        "n": n,
        "h": h,
        "alg": alg,
        "sym": sym,
        "run": run_num,
    }

    runner = {
        "IBIS": lambda: _run_ibis(territory, n, h, sym, timeout_ms),
        "CEGAR": lambda: _run_cegar(territory, n, h, sym, timeout_ms),
        "Q-IBIS": lambda: _run_qibis(territory, n, h, sym, timeout_ms),
    }[alg]

    # Set alarm for wall-clock timeout
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)

    try:
        t0 = time.perf_counter()
        result = runner()
        elapsed = round(time.perf_counter() - t0, 3)

        signal.alarm(0)  # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)

        return {
            **row_base,
            "status": "ok",
            "ne": result["ne"],
            "payoff": result["payoff"],
            "iters": result["iters"],
            "time_s": elapsed,
            "reason": result["reason"],
            "cegar_final_blocks": result["cegar_final_blocks"],
        }

    except TimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {
            **row_base,
            "status": "timeout",
            "ne": None,
            "payoff": None,
            "iters": None,
            "time_s": timeout_s,
            "reason": None,
            "cegar_final_blocks": None,
        }

    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        elapsed = round(time.perf_counter() - t0, 3)
        return {
            **row_base,
            "status": "error",
            "ne": None,
            "payoff": None,
            "iters": None,
            "time_s": elapsed,
            "reason": str(e),
            "cegar_final_blocks": None,
        }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_all(
    num_runs: int = 3,
    timeout_s: int = 300,
    filter_pattern: str | None = None,
    output_file: str | None = None,
    include_qibis: bool = False,
) -> None:
    # Auto-discover grids
    grid_infos = discover_configs()

    # Apply filter
    if filter_pattern:
        pat = re.compile(filter_pattern)
        grid_infos = [(n, gt, p, S) for n, gt, p, S in grid_infos if pat.search(n)]

    # Build full config list: (grid_name, grid_type, path, S, n, h)
    configs: list[tuple[str, str, str, int, int, int]] = []
    for name, grid_type, path, S in grid_infos:
        for n in feasible_agent_counts(S):
            h = S // n
            configs.append((name, grid_type, path, S, n, h))

    # Sort by (S, n, grid_type, name) for predictable ordering
    configs.sort(key=lambda c: (c[3], c[4], c[1], c[0]))

    # Algorithms to run
    algorithms = ["IBIS", "CEGAR"]
    if include_qibis:
        algorithms.append("Q-IBIS")

    total_runs = len(configs) * len(algorithms) * 2 * num_runs  # x2 for sym True/False
    print(
        f"Discovered {len(grid_infos)} grids, {len(configs)} configs",
        file=sys.stderr,
    )
    print(
        f"Benchmark: {len(configs)} configs x {len(algorithms)} algs x 2 sym x {num_runs} runs = {total_runs} total",
        file=sys.stderr,
    )

    # Output destination
    out = open(output_file, "w") if output_file else sys.stdout

    results: list[dict] = []
    completed = 0

    try:
        for run in range(1, num_runs + 1):
            print(f"\n=== Run {run}/{num_runs} ===", file=sys.stderr)
            for grid_name, grid_type, path, S, n, h in configs:
                territory = load_grid(path)

                for alg in algorithms:
                    for sym in [False, True]:
                        completed += 1
                        gt = "sym" if grid_type == "symmetric" else "asym"
                        label = f"[{completed}/{total_runs}] {grid_name}({gt}) S={S} n={n} h={h} {alg} sym={sym}"
                        print(f"  {label} ...", end="", file=sys.stderr, flush=True)

                        row = run_single(
                            territory=territory,
                            grid_name=grid_name,
                            grid_type=grid_type,
                            S=S,
                            n=n,
                            h=h,
                            alg=alg,
                            sym=sym,
                            timeout_s=timeout_s,
                            run_num=run,
                        )
                        results.append(row)
                        print(json.dumps(row), file=out, flush=True)

                        status = row["status"]
                        t = row["time_s"]
                        print(f" {status} ({t:.1f}s)", file=sys.stderr)

    finally:
        if output_file:
            out.close()

    # Print summary
    _print_summary(results)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    groups: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {"iters": [], "time": [], "ne": [], "status": []}
    )

    for d in results:
        key = (d["grid"], d["grid_type"], d["S"], d["n"], d["h"], d["alg"], d["sym"])
        groups[key]["status"].append(d["status"])
        if d["status"] == "ok":
            groups[key]["iters"].append(d["iters"])
            groups[key]["time"].append(d["time_s"])
            groups[key]["ne"].append(d["ne"])

    print("\n=== Median Results ===", file=sys.stderr)
    header = (
        f"{'Grid':>12s} {'Type':>5s} {'S':>3s} {'n':>2s} {'h':>3s} "
        f"{'Alg':>8s} {'Sym':>5s} {'NE':>4s} {'Iters':>5s} {'Time':>9s} {'Status':>8s}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for key in sorted(groups.keys()):
        grid, gtype, S, n, h, alg, sym = key
        g = groups[key]

        statuses = g["status"]
        if all(s == "timeout" for s in statuses):
            status_str = "timeout"
        elif all(s == "error" for s in statuses):
            status_str = "error"
        elif any(s == "ok" for s in statuses):
            status_str = "ok"
        else:
            status_str = statuses[0]

        if g["time"]:
            med_time = median(g["time"])
            time_str = f"{med_time:>8.2f}s"
        else:
            time_str = f"{'—':>9s}"

        if g["iters"]:
            med_iters = int(median(g["iters"]))
            iters_str = f"{med_iters:>5d}"
        else:
            iters_str = f"{'—':>5s}"

        if g["ne"]:
            ne_str = "Yes" if any(g["ne"]) else "No"
        else:
            ne_str = "—"

        gtype_short = "sym" if gtype == "symmetric" else "asym"
        print(
            f"{grid:>12s} {gtype_short:>5s} {S:>3d} {n:>2d} {h:>3d} "
            f"{alg:>8s} {str(sym):>5s} {ne_str:>4s} {iters_str} {time_str} {status_str:>8s}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of repeated runs for median aggregation (default: 3)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-config wall-clock timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Regex to filter configs by grid name (e.g. '4x4|6x4')",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write JSON lines to file instead of stdout",
    )
    parser.add_argument(
        "--include-qibis", action="store_true",
        help="Include Q-IBIS algorithm in the benchmark",
    )
    args = parser.parse_args()

    run_all(
        num_runs=args.runs,
        timeout_s=args.timeout,
        filter_pattern=args.filter,
        output_file=args.output,
        include_qibis=args.include_qibis,
    )
