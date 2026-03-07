"""Benchmark runner: executes scenarios and collects results.

Outputs one JSON line per (scenario, run) to the output file.
Uses SIGALRM for wall-clock timeouts on Unix.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from rcmas.model import Territory

from .scenarios import Scenario


@dataclass(frozen=True, slots=True)
class Result:
    """One benchmark measurement."""

    suite: str
    name: str
    grid_path: str
    num_sectors: int
    num_agents: int
    horizon: int
    algorithm: str
    symmetry: bool
    partition: str
    run: int

    status: str  # "ok", "timeout", "error"
    found_ne: bool | None
    payoff: list[int] | None
    iterations: int | None
    time_s: float
    reason: str | None
    cegar_final_blocks: int | None


class _Timeout(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _Timeout()


# ---------------------------------------------------------------------------
# Algorithm dispatch
# ---------------------------------------------------------------------------

def _run_algorithm(
    territory: Territory,
    scenario: Scenario,
    timeout_ms: int | None,
) -> dict[str, Any]:
    """Run a single algorithm and return raw result dict."""
    alg = scenario.algorithm
    n = scenario.num_agents
    h = scenario.horizon
    sym = scenario.symmetry
    max_iters = 30

    if alg == "ibis":
        from rcmas.ibis import solve_ibis

        res = solve_ibis(
            territory=territory, num_agents=n, horizon=h,
            max_iters=max_iters, symmetry=sym, timeout_ms=timeout_ms,
        )
        return {
            "found_ne": res.found_ne,
            "payoff": list(res.payoff_by_agent) if res.payoff_by_agent else None,
            "iterations": res.iterations,
            "reason": res.reason,
            "cegar_final_blocks": None,
        }

    elif alg == "qibis":
        from rcmas.qibis import QibisConfig, solve_qibis

        cfg = QibisConfig(max_iters=max_iters, symmetry=sym, timeout_ms=timeout_ms)
        res = solve_qibis(territory=territory, num_agents=n, horizon=h, cfg=cfg)
        return {
            "found_ne": res.found_ne,
            "payoff": list(res.payoff_by_agent) if res.payoff_by_agent else None,
            "iterations": res.iterations,
            "reason": res.reason,
            "cegar_final_blocks": None,
        }

    elif alg in ("cegar-ibis", "cegar-qibis"):
        from rcmas.cegar import solve_cegar

        synthesiser = "ibis" if alg == "cegar-ibis" else "qibis"
        res = solve_cegar(
            territory=territory, num_agents=n, horizon=h,
            initial_partition=scenario.partition,
            synthesiser=synthesiser,
            max_iters=max_iters, symmetry=sym, timeout_ms=timeout_ms,
        )
        return {
            "found_ne": res.found_ne,
            "payoff": list(res.payoff_by_agent) if res.payoff_by_agent else None,
            "iterations": res.iterations,
            "reason": res.reason,
            "cegar_final_blocks": res.final_partition_size,
        }

    else:
        raise ValueError(f"unknown algorithm: {alg}")


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: Scenario,
    run_num: int,
    timeout_s: int,
) -> Result:
    """Execute a single scenario and return a Result."""
    with open(scenario.grid_path) as f:
        territory = Territory.from_ascii(f)

    S = len(territory.sectors)
    timeout_ms = timeout_s * 1000

    base = dict(
        suite=scenario.suite,
        name=scenario.name,
        grid_path=scenario.grid_path,
        num_sectors=S,
        num_agents=scenario.num_agents,
        horizon=scenario.horizon,
        algorithm=scenario.algorithm,
        symmetry=scenario.symmetry,
        partition=scenario.partition,
        run=run_num,
    )

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    t0 = time.perf_counter()

    try:
        raw = _run_algorithm(territory, scenario, timeout_ms)
        elapsed = round(time.perf_counter() - t0, 3)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        return Result(
            **base,
            status="ok",
            found_ne=raw["found_ne"],
            payoff=raw["payoff"],
            iterations=raw["iterations"],
            time_s=elapsed,
            reason=raw["reason"],
            cegar_final_blocks=raw["cegar_final_blocks"],
        )

    except _Timeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return Result(
            **base, status="timeout", found_ne=None, payoff=None,
            iterations=None, time_s=float(timeout_s), reason="timeout",
            cegar_final_blocks=None,
        )

    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 3)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return Result(
            **base, status="error", found_ne=None, payoff=None,
            iterations=None, time_s=elapsed, reason=str(e),
            cegar_final_blocks=None,
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    scenarios: list[Scenario],
    *,
    num_runs: int = 3,
    timeout_s: int = 300,
    output_path: str | None = None,
    progress: bool = True,
) -> list[Result]:
    """Run all scenarios and collect results.

    Results are written incrementally as JSON lines to output_path (if given)
    and also returned as a list.
    """
    total = len(scenarios) * num_runs
    results: list[Result] = []

    out = open(output_path, "w") if output_path else None

    try:
        idx = 0
        for run_num in range(1, num_runs + 1):
            if progress:
                print(f"\n=== Run {run_num}/{num_runs} ===", file=sys.stderr)

            for scenario in scenarios:
                idx += 1
                if progress:
                    label = (
                        f"[{idx}/{total}] {scenario.suite} "
                        f"{scenario.name}"
                    )
                    print(f"  {label} ...", end="", file=sys.stderr, flush=True)

                result = run_scenario(scenario, run_num, timeout_s)
                results.append(result)

                if out is not None:
                    out.write(json.dumps(asdict(result)) + "\n")
                    out.flush()

                if progress:
                    status = result.status
                    t = result.time_s
                    ne = result.found_ne
                    ne_str = f" NE={ne}" if ne is not None else ""
                    print(f" {status} ({t:.1f}s){ne_str}", file=sys.stderr)

    finally:
        if out is not None:
            out.close()

    return results
