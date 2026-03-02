#!/usr/bin/env python3
"""Reproduce the experimental results for the EUMAS 2026 paper.

Usage:
    python run_experiments.py [--runs N]

Requires the rcmas package to be installed (pip install -e .).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from statistics import median

from rcmas.ibis import solve_ibis
from rcmas.model import Territory
from rcmas.qibis import QibisConfig, solve_qibis

CONFIGS = [
    ("5x5", 3, 8),
    # ("2x2", 2, 2),
    # ("3x3", 3, 3),
    # ("4x4", 2, 8),
    # ("4x4", 4, 4),
]

GRIDS_DIR = "./grids/symmetric"


def load_grid(name: str) -> Territory:
    with open(f"{GRIDS_DIR}/{name}.txt") as f:
        return Territory.from_ascii(f)


def run_all(num_runs: int = 3) -> None:
    results: list[dict] = []

    for run in range(1, num_runs + 1):
        print(f"=== Run {run}/{num_runs} ===", file=sys.stderr)
        for grid_name, A, H in CONFIGS:
            territory = load_grid(grid_name)
            S = len(territory.sectors)

            for sym in [False, True]:
                # IBIS
                t0 = time.time()
                ibis_result = solve_ibis(
                    territory=territory,
                    num_agents=A,
                    horizon=H,
                    max_iters=30,
                    symmetry=sym,
                )
                elapsed = round(time.time() - t0, 3)
                row = {
                    "grid": grid_name,
                    "A": A,
                    "H": H,
                    "S": S,
                    "alg": "IBIS",
                    "sym": sym,
                    "ne": ibis_result.found_ne,
                    "iters": ibis_result.iterations,
                    "time": elapsed,
                    "run": run,
                }
                results.append(row)
                print(json.dumps(row), flush=True)

                # Q-IBIS
                cfg = QibisConfig(max_iters=30, symmetry=sym)
                t0 = time.time()
                qibis_result = solve_qibis(
                    territory=territory, num_agents=A, horizon=H, cfg=cfg
                )
                elapsed = round(time.time() - t0, 3)
                row = {
                    "grid": grid_name,
                    "A": A,
                    "H": H,
                    "S": S,
                    "alg": "Q-IBIS",
                    "sym": sym,
                    "ne": qibis_result.found_ne,
                    "iters": qibis_result.iterations,
                    "time": elapsed,
                    "run": run,
                }
                results.append(row)
                print(json.dumps(row), flush=True)

    # Compute medians
    print("\n=== Median Results ===", file=sys.stderr)
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: {"iters": [], "time": []})
    for d in results:
        key = (d["grid"], d["A"], d["H"], d["alg"], d["sym"])
        groups[key]["iters"].append(d["iters"])
        groups[key]["time"].append(d["time"])

    print(
        f"{'Grid':>5s} {'A':>2s} {'H':>2s} {'Alg':>8s} {'Sym':>5s} {'Iters':>5s} {'Time':>8s}",
        file=sys.stderr,
    )
    for key in sorted(groups.keys()):
        g, a, h, alg, sym = key
        med_iters = int(median(groups[key]["iters"]))
        med_time = round(median(groups[key]["time"]), 2)
        print(
            f"{g:>5s} {a:>2d} {h:>2d} {alg:>8s} {str(sym):>5s} {med_iters:>5d} {med_time:>8.2f}s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    args = parser.parse_args()
    run_all(args.runs)
