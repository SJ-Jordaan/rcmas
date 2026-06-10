"""Benchmark runner: executes scenarios and collects results.

Outputs one JSON line per (scenario, run) to the output file.
Uses SIGALRM for wall-clock timeouts on Unix.

Supports resumption: when *resume=True*, existing results are loaded from the
output file and only missing (scenario, run) pairs are executed.  New results
are appended rather than overwriting.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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


# Raising from a signal handler is unreliable around Z3's ctypes layer: the
# exception can be swallowed (inside a ctypes callback) or wrapped as
# ctypes.ArgumentError (during argument conversion). The re-arm gives a
# swallowed raise another chance to land at a safe bytecode boundary, and
# _is_timeout_error recovers the wrapped case.
_REARM_INTERVAL_S = 5


def _alarm_handler(signum: int, frame: Any) -> None:
    signal.alarm(_REARM_INTERVAL_S)
    raise _Timeout()


def _is_timeout_error(e: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, _Timeout):
            return True
        cur = cur.__cause__ or cur.__context__
    return "_Timeout" in str(e)


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

    elif alg == "sibis":
        from rcmas.ibis import solve_sibis

        res = solve_sibis(
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

    elif alg in ("cegar-ibis", "cegar-qibis", "cegar-sibis"):
        from rcmas.cegar import solve_cegar

        synthesiser = {"cegar-ibis": "ibis", "cegar-qibis": "qibis", "cegar-sibis": "sibis"}[alg]
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

        if elapsed >= timeout_s:
            # The alarm's exception was swallowed and the run overran its
            # budget; count it as a timeout like every other over-budget run.
            return Result(
                **base, status="timeout", found_ne=None, payoff=None,
                iterations=None, time_s=float(timeout_s), reason="timeout",
                cegar_final_blocks=None,
            )

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
        if _is_timeout_error(e):
            return Result(
                **base, status="timeout", found_ne=None, payoff=None,
                iterations=None, time_s=float(timeout_s), reason="timeout",
                cegar_final_blocks=None,
            )
        return Result(
            **base, status="error", found_ne=None, payoff=None,
            iterations=None, time_s=elapsed, reason=str(e),
            cegar_final_blocks=None,
        )


# ---------------------------------------------------------------------------
# Parallel helpers
# ---------------------------------------------------------------------------

def _pool_init() -> None:
    """Ignore SIGINT in pool workers; the main process handles shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_scenario_task(args: tuple) -> Result:
    """Picklable wrapper for multiprocessing dispatch."""
    scenario, run_num, timeout_s = args
    return run_scenario(scenario, run_num, timeout_s)


def _print_result(result: Result) -> None:
    ne_str = f" NE={result.found_ne}" if result.found_ne is not None else ""
    print(f" {result.status} ({result.time_s:.1f}s){ne_str}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _result_key(row: dict[str, Any]) -> tuple:
    """Return the unique key that identifies a (scenario, run) pair."""
    return (row["suite"], row["name"], row["algorithm"], row["symmetry"],
            row.get("partition", "orbit"), row["run"])


def _load_completed(path: str) -> tuple[set[tuple], list[dict]]:
    """Load existing results from a JSONL file.

    Returns (completed_keys, rows) where *completed_keys* is the set of
    ``_result_key`` tuples already present and *rows* is the raw list of
    parsed dicts (so callers can convert to Result/ResultRow).
    """
    completed: set[tuple] = set()
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return completed, rows
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            completed.add(_result_key(row))
    return completed, rows


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
    workers: int = 1,
    resume: bool = False,
) -> list[Result]:
    """Run all scenarios and collect results.

    Results are written incrementally as JSON lines to output_path (if given)
    and also returned as a list.

    When *resume* is True and *output_path* points to an existing file, any
    (scenario, run) pairs already present in the file are skipped and new
    results are appended.

    When *workers* > 1, scenarios are dispatched across that many processes
    via ``multiprocessing.Pool``.  Each worker uses its own SIGALRM for
    per-scenario timeouts, so results are identical to sequential execution.
    """
    if workers <= 0:
        workers = os.cpu_count() or 1

    # --- Resume: load existing results and filter work items ---------------
    completed_keys: set[tuple] = set()
    prior_results: list[Result] = []

    if resume and output_path:
        completed_keys, prior_rows = _load_completed(output_path)
        if completed_keys:
            prior_results = [Result(**row) for row in prior_rows]
            if progress:
                print(
                    f"Resuming: {len(completed_keys)} results loaded, "
                    f"skipping completed experiments.",
                    file=sys.stderr,
                )

    total_all = len(scenarios) * num_runs
    total_skip = 0

    # Build filtered work list
    filtered_scenarios: list[tuple[Scenario, int]] = []
    for run_num in range(1, num_runs + 1):
        for scenario in scenarios:
            key = (scenario.suite, scenario.name, scenario.algorithm,
                   scenario.symmetry, scenario.partition, run_num)
            if key in completed_keys:
                total_skip += 1
            else:
                filtered_scenarios.append((scenario, run_num))

    if progress and total_skip > 0:
        print(
            f"Skipping {total_skip}/{total_all} already completed. "
            f"Running {len(filtered_scenarios)} remaining.",
            file=sys.stderr,
        )

    if not filtered_scenarios:
        if progress:
            print("All experiments already completed.", file=sys.stderr)
        return prior_results

    total = len(filtered_scenarios)
    results: list[Result] = []

    # Open in append mode when resuming, write mode otherwise
    if output_path:
        mode = "a" if resume and completed_keys else "w"
        out = open(output_path, mode)
    else:
        out = None

    try:
        if workers == 1:
            _run_sequential_filtered(filtered_scenarios, timeout_s, total, results, out, progress)
        else:
            _run_parallel_filtered(filtered_scenarios, timeout_s, total, results, out, progress, workers)
    finally:
        if out is not None:
            out.close()

    return prior_results + results


def _run_sequential(
    scenarios: list[Scenario],
    num_runs: int,
    timeout_s: int,
    total: int,
    results: list[Result],
    out: Any,
    progress: bool,
) -> None:
    idx = 0
    for run_num in range(1, num_runs + 1):
        if progress:
            print(f"\n=== Run {run_num}/{num_runs} ===", file=sys.stderr)

        for scenario in scenarios:
            idx += 1
            if progress:
                label = f"[{idx}/{total}] {scenario.suite} {scenario.name}"
                print(f"  {label} ...", end="", file=sys.stderr, flush=True)

            result = run_scenario(scenario, run_num, timeout_s)
            results.append(result)

            if out is not None:
                out.write(json.dumps(asdict(result)) + "\n")
                out.flush()

            if progress:
                _print_result(result)


def _run_sequential_filtered(
    work: list[tuple[Scenario, int]],
    timeout_s: int,
    total: int,
    results: list[Result],
    out: Any,
    progress: bool,
) -> None:
    for idx, (scenario, run_num) in enumerate(work, 1):
        if progress:
            label = f"[{idx}/{total}] {scenario.suite} {scenario.name} r{run_num}"
            print(f"  {label} ...", end="", file=sys.stderr, flush=True)

        result = run_scenario(scenario, run_num, timeout_s)
        results.append(result)

        if out is not None:
            out.write(json.dumps(asdict(result)) + "\n")
            out.flush()

        if progress:
            _print_result(result)


def _run_parallel(
    scenarios: list[Scenario],
    num_runs: int,
    timeout_s: int,
    total: int,
    results: list[Result],
    out: Any,
    progress: bool,
    workers: int,
) -> None:
    import multiprocessing

    work = [
        (scenario, run_num, timeout_s)
        for run_num in range(1, num_runs + 1)
        for scenario in scenarios
    ]

    if progress:
        print(f"\nParallel: {workers} workers, {total} tasks", file=sys.stderr)

    with multiprocessing.Pool(processes=workers, initializer=_pool_init) as pool:
        for idx, result in enumerate(pool.imap_unordered(_run_scenario_task, work), 1):
            results.append(result)

            if out is not None:
                out.write(json.dumps(asdict(result)) + "\n")
                out.flush()

            if progress:
                label = f"[{idx}/{total}] {result.suite} {result.name} r{result.run}"
                print(f"  {label} ...", end="", file=sys.stderr, flush=True)
                _print_result(result)


def _run_parallel_filtered(
    work: list[tuple[Scenario, int]],
    timeout_s: int,
    total: int,
    results: list[Result],
    out: Any,
    progress: bool,
    workers: int,
) -> None:
    import multiprocessing

    pool_work = [(scenario, run_num, timeout_s) for scenario, run_num in work]

    if progress:
        print(f"\nParallel: {workers} workers, {total} tasks", file=sys.stderr)

    with multiprocessing.Pool(processes=workers, initializer=_pool_init) as pool:
        for idx, result in enumerate(pool.imap_unordered(_run_scenario_task, pool_work), 1):
            results.append(result)

            if out is not None:
                out.write(json.dumps(asdict(result)) + "\n")
                out.flush()

            if progress:
                label = f"[{idx}/{total}] {result.suite} {result.name} r{result.run}"
                print(f"  {label} ...", end="", file=sys.stderr, flush=True)
                _print_result(result)
