#!/usr/bin/env python3
"""Long-running symmetry experiment on large grids.

Runs 8x8 (8 agents) and 6x6 (6 agents) with sym ON/OFF in parallel.
All 4 configs run simultaneously — wall clock ≈ 1 hour max.
Each config gets its own verbose log file in logs/.

Usage:
    cd tools/rcmas
    source .venv/bin/activate
    nohup python run_large_symmetry.py &

    # Watch progress:
    tail -f logs/8x8_n8_ibis_symON.log
    tail -f logs/*.log   # all at once
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRIDS = os.path.join(SCRIPT_DIR, "grids", "symmetric")
OUTPUT = os.path.join(SCRIPT_DIR, "results-large-symmetry.jsonl")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
TIMEOUT_S = 3600  # 1 hour per config


@dataclass(frozen=True)
class Config:
    name: str
    grid_path: str
    num_agents: int
    horizon: int
    algorithm: str
    symmetry: bool


CONFIGS = [
    # 8x8, 8 agents, h=8 — the main event
    Config("8x8_n8_ibis_symOFF", os.path.join(GRIDS, "8x8.txt"), 8, 8, "ibis", False),
    Config("8x8_n8_ibis_symON",  os.path.join(GRIDS, "8x8.txt"), 8, 8, "ibis", True),
    # 6x6, 6 agents, h=6 — calibration
    Config("6x6_n6_ibis_symOFF", os.path.join(GRIDS, "6x6.txt"), 6, 6, "ibis", False),
    Config("6x6_n6_ibis_symON",  os.path.join(GRIDS, "6x6.txt"), 6, 6, "ibis", True),
]


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_config(cfg: Config) -> dict[str, Any]:
    """Run a single config with verbose logging to its own log file."""
    log_path = os.path.join(LOG_DIR, f"{cfg.name}.log")
    log_file = open(log_path, "w")

    # Redirect stderr to the log file so progress/timing output goes there
    old_stderr = sys.stderr
    sys.stderr = log_file

    def log(msg: str):
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"{'='*60}")
    log(f"Config: {cfg.name}")
    log(f"Grid:   {cfg.grid_path}")
    log(f"Agents: {cfg.num_agents}, Horizon: {cfg.horizon}")
    log(f"Algo:   {cfg.algorithm}, Symmetry: {cfg.symmetry}")
    log(f"Timeout: {TIMEOUT_S}s")
    log(f"Start:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")

    from rcmas.model import Territory

    with open(cfg.grid_path) as f:
        territory = Territory.from_ascii(f)

    S = len(territory.sectors)
    timeout_ms = TIMEOUT_S * 1000
    max_iters = 30

    # Set up wall-clock alarm
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_S)
    t0 = time.perf_counter()

    result = {
        "name": cfg.name,
        "grid": os.path.basename(cfg.grid_path),
        "num_sectors": S,
        "num_agents": cfg.num_agents,
        "horizon": cfg.horizon,
        "algorithm": cfg.algorithm,
        "symmetry": cfg.symmetry,
    }

    try:
        if cfg.algorithm == "ibis":
            from rcmas.ibis import solve_ibis
            res = solve_ibis(
                territory=territory,
                num_agents=cfg.num_agents,
                horizon=cfg.horizon,
                max_iters=max_iters,
                symmetry=cfg.symmetry,
                timeout_ms=timeout_ms,
                progress=True,
                timing=True,
            )
        elif cfg.algorithm == "cegar-ibis":
            from rcmas.cegar import solve_cegar
            res = solve_cegar(
                territory=territory,
                num_agents=cfg.num_agents,
                horizon=cfg.horizon,
                initial_partition="orbit",
                synthesiser="ibis",
                max_iters=max_iters,
                symmetry=cfg.symmetry,
                timeout_ms=timeout_ms,
                progress=True,
                timing=True,
            )
        else:
            raise ValueError(f"unsupported algorithm: {cfg.algorithm}")

        elapsed = round(time.perf_counter() - t0, 3)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        result.update({
            "status": "ok",
            "found_ne": res.found_ne,
            "payoff": list(res.payoff_by_agent) if res.payoff_by_agent else None,
            "iterations": res.iterations,
            "time_s": elapsed,
            "reason": res.reason,
        })

        log(f"\n{'='*60}")
        log(f"FINISHED: {'NE found' if res.found_ne else 'no NE'}")
        log(f"Payoff:   {result['payoff']}")
        log(f"Iters:    {res.iterations}")
        log(f"Time:     {elapsed:.3f}s")
        log(f"Reason:   {res.reason}")
        log(f"End:      {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"{'='*60}")

    except _Timeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        elapsed = round(time.perf_counter() - t0, 3)

        result.update({
            "status": "timeout",
            "found_ne": None,
            "payoff": None,
            "iterations": None,
            "time_s": elapsed,
            "reason": "timeout",
        })

        log(f"\n{'='*60}")
        log(f"TIMEOUT after {elapsed:.1f}s")
        log(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"{'='*60}")

    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        elapsed = round(time.perf_counter() - t0, 3)

        result.update({
            "status": "error",
            "found_ne": None,
            "payoff": None,
            "iterations": None,
            "time_s": elapsed,
            "reason": str(e),
        })

        log(f"\n{'='*60}")
        log(f"ERROR after {elapsed:.1f}s: {e}")
        log(f"{'='*60}")

    finally:
        sys.stderr = old_stderr
        log_file.close()

    # Print completion to main stderr
    status_str = result["status"]
    if status_str == "ok":
        ne = "NE" if result["found_ne"] else "no NE"
        print(f"[DONE] {cfg.name}: {ne} in {result['time_s']:.1f}s "
              f"(iters={result['iterations']}, payoff={result['payoff']})",
              file=sys.stderr, flush=True)
    else:
        print(f"[DONE] {cfg.name}: {status_str} after {result['time_s']:.1f}s",
              file=sys.stderr, flush=True)

    return result


def _run_wrapper(cfg: Config) -> dict[str, Any]:
    return _run_config(cfg)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Running {len(CONFIGS)} configs in parallel, timeout={TIMEOUT_S}s each", flush=True)
    print(f"Results: {OUTPUT}", flush=True)
    print(f"Logs:    {LOG_DIR}/", flush=True)
    print(f"Start:   {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(flush=True)

    for c in CONFIGS:
        print(f"  {c.name:30s}  {os.path.basename(c.grid_path):8s}  "
              f"A={c.num_agents} h={c.horizon} sym={c.symmetry}", flush=True)
    print(flush=True)
    print(f"Watch progress:  tail -f {LOG_DIR}/<name>.log", flush=True)
    print(f"Watch all:       tail -f {LOG_DIR}/*.log", flush=True)
    print(flush=True)

    with open(OUTPUT, "w") as out:
        with Pool(processes=len(CONFIGS), initializer=_worker_init) as pool:
            for result in pool.imap_unordered(_run_wrapper, CONFIGS):
                out.write(json.dumps(result) + "\n")
                out.flush()

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    with open(OUTPUT) as f:
        rows = [json.loads(line) for line in f]
    for r in sorted(rows, key=lambda x: x["name"]):
        if r["status"] == "ok":
            ne = "NE" if r["found_ne"] else "no NE"
            print(f"  {r['name']:30s}  {ne:>6s}  {r['time_s']:8.1f}s  "
                  f"iters={r['iterations']}  payoff={r['payoff']}", flush=True)
        else:
            print(f"  {r['name']:30s}  {r['status']:>6s}  {r['time_s']:8.1f}s  "
                  f"reason={r['reason']}", flush=True)


if __name__ == "__main__":
    main()
