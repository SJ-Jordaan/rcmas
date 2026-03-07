"""Results analysis and reporting.

Reads JSON-lines result files and produces summary tables showing:
- Speedup from symmetry breaking (per suite)
- Algorithm comparison (IBIS vs Q-IBIS vs CEGAR variants)
- Scalability curves
- Success rates
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median


@dataclass
class ResultRow:
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
    status: str
    found_ne: bool | None
    payoff: list[int] | None
    iterations: int | None
    time_s: float
    reason: str | None
    cegar_final_blocks: int | None


def load_results(path: str) -> list[ResultRow]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(ResultRow(**d))
    return rows


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _group_key_base(r: ResultRow) -> tuple:
    """Group key excluding algorithm/symmetry (for paired comparisons)."""
    return (r.suite, r.num_sectors, r.num_agents, r.horizon, r.grid_path)


def _median_time(rows: list[ResultRow]) -> float | None:
    times = [r.time_s for r in rows if r.status == "ok"]
    return median(times) if times else None


def _median_iters(rows: list[ResultRow]) -> int | None:
    iters = [r.iterations for r in rows if r.status == "ok" and r.iterations is not None]
    return int(median(iters)) if iters else None


def _ne_rate(rows: list[ResultRow]) -> float:
    ok_rows = [r for r in rows if r.status == "ok"]
    if not ok_rows:
        return 0.0
    return sum(1 for r in ok_rows if r.found_ne) / len(ok_rows)


def _timeout_rate(rows: list[ResultRow]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.status == "timeout") / len(rows)


# ---------------------------------------------------------------------------
# E1/E2/E3: Symmetry speedup analysis
# ---------------------------------------------------------------------------

def symmetry_speedup_table(
    results: list[ResultRow],
    suites: list[str] | None = None,
) -> None:
    """Print speedup from symmetry=True vs symmetry=False per config.

    Groups by (grid, agents, horizon, algorithm) and computes
    median_time(sym=False) / median_time(sym=True).
    """
    if suites:
        results = [r for r in results if r.suite in suites]

    # Group by (suite, grid, S, n, h, algorithm)
    groups: dict[tuple, dict[bool, list[ResultRow]]] = defaultdict(lambda: {True: [], False: []})
    for r in results:
        key = (r.suite, r.grid_path.split("/")[-1], r.num_sectors, r.num_agents, r.horizon, r.algorithm)
        groups[key][r.symmetry].append(r)

    print("\n=== Symmetry Speedup ===", file=sys.stderr)
    header = f"{'Suite':>5s} {'Grid':>20s} {'S':>3s} {'n':>2s} {'h':>3s} {'Alg':>12s} {'T(off)':>8s} {'T(on)':>8s} {'Speedup':>8s} {'NE(off)':>7s} {'NE(on)':>7s}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for key in sorted(groups.keys()):
        suite, grid, S, n, h, alg = key
        sym_off = groups[key][False]
        sym_on = groups[key][True]

        t_off = _median_time(sym_off)
        t_on = _median_time(sym_on)

        t_off_str = f"{t_off:.2f}s" if t_off is not None else "TO"
        t_on_str = f"{t_on:.2f}s" if t_on is not None else "TO"

        if t_off is not None and t_on is not None and t_on > 0:
            speedup = t_off / t_on
            sp_str = f"{speedup:.2f}x"
        else:
            sp_str = "—"

        ne_off_str = f"{_ne_rate(sym_off):.0%}" if sym_off else "—"
        ne_on_str = f"{_ne_rate(sym_on):.0%}" if sym_on else "—"

        print(
            f"{suite:>5s} {grid:>20s} {S:>3d} {n:>2d} {h:>3d} {alg:>12s} "
            f"{t_off_str:>8s} {t_on_str:>8s} {sp_str:>8s} {ne_off_str:>7s} {ne_on_str:>7s}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# E4/E5: Algorithm comparison
# ---------------------------------------------------------------------------

def algorithm_comparison_table(
    results: list[ResultRow],
    suites: list[str] | None = None,
) -> None:
    """Compare algorithms on the same grid/agent configs."""
    if suites:
        results = [r for r in results if r.suite in suites]

    # Group by (grid, S, n, h) then by algorithm
    groups: dict[tuple, dict[str, list[ResultRow]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        base_key = (r.grid_path.split("/")[-1], r.num_sectors, r.num_agents, r.horizon)
        alg_key = r.algorithm
        if r.algorithm.startswith("cegar"):
            alg_key = f"{r.algorithm}({r.partition})"
        groups[base_key][alg_key].append(r)

    print("\n=== Algorithm Comparison ===", file=sys.stderr)
    header = f"{'Grid':>20s} {'S':>3s} {'n':>2s} {'h':>3s} {'Algorithm':>20s} {'Time':>8s} {'Iters':>5s} {'NE%':>5s} {'TO%':>5s} {'Blocks':>6s}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for base_key in sorted(groups.keys()):
        grid, S, n, h = base_key
        for alg_key in sorted(groups[base_key].keys()):
            rows = groups[base_key][alg_key]
            t = _median_time(rows)
            iters = _median_iters(rows)

            t_str = f"{t:.2f}s" if t is not None else "TO"
            i_str = str(iters) if iters is not None else "—"
            ne_str = f"{_ne_rate(rows):.0%}"
            to_str = f"{_timeout_rate(rows):.0%}"

            blocks_list = [r.cegar_final_blocks for r in rows if r.cegar_final_blocks is not None]
            b_str = str(int(median(blocks_list))) if blocks_list else "—"

            print(
                f"{grid:>20s} {S:>3d} {n:>2d} {h:>3d} {alg_key:>20s} "
                f"{t_str:>8s} {i_str:>5s} {ne_str:>5s} {to_str:>5s} {b_str:>6s}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# E7: Scalability
# ---------------------------------------------------------------------------

def scalability_table(
    results: list[ResultRow],
) -> None:
    """Show how solve time grows with grid size."""
    results = [r for r in results if r.suite == "E7"]
    if not results:
        print("\n(No E7 results)", file=sys.stderr)
        return

    # Group by (algorithm, symmetry) then by S
    groups: dict[tuple, dict[int, list[ResultRow]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        groups[(r.algorithm, r.symmetry)][r.num_sectors].append(r)

    print("\n=== Scalability (E7) ===", file=sys.stderr)
    header = f"{'Algorithm':>12s} {'Sym':>5s} " + " ".join(f"{'S=' + str(s):>8s}" for s in sorted({r.num_sectors for r in results}))
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    sizes = sorted({r.num_sectors for r in results})
    for (alg, sym) in sorted(groups.keys()):
        cells = []
        for s in sizes:
            rows = groups[(alg, sym)].get(s, [])
            t = _median_time(rows)
            if t is not None:
                cells.append(f"{t:.2f}s")
            elif any(r.status == "timeout" for r in rows):
                cells.append("TO")
            else:
                cells.append("—")
        row_str = " ".join(f"{c:>8s}" for c in cells)
        print(f"{alg:>12s} {str(sym):>5s} {row_str}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def suite_summary(results: list[ResultRow]) -> None:
    """Print per-suite summary statistics."""
    suites: dict[str, list[ResultRow]] = defaultdict(list)
    for r in results:
        suites[r.suite].append(r)

    print("\n=== Suite Summary ===", file=sys.stderr)
    header = f"{'Suite':>5s} {'Scenarios':>10s} {'OK':>5s} {'TO':>5s} {'Err':>5s} {'NE Found':>9s} {'Med Time':>9s}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for suite in sorted(suites.keys()):
        rows = suites[suite]
        n_total = len(rows)
        n_ok = sum(1 for r in rows if r.status == "ok")
        n_to = sum(1 for r in rows if r.status == "timeout")
        n_err = sum(1 for r in rows if r.status == "error")
        ne_found = sum(1 for r in rows if r.status == "ok" and r.found_ne)
        t = _median_time(rows)
        t_str = f"{t:.2f}s" if t is not None else "—"

        print(
            f"{suite:>5s} {n_total:>10d} {n_ok:>5d} {n_to:>5d} {n_err:>5d} "
            f"{ne_found:>9d} {t_str:>9s}",
            file=sys.stderr,
        )


def full_report(results: list[ResultRow]) -> None:
    """Generate the complete analysis report."""
    suite_summary(results)
    symmetry_speedup_table(results, suites=["E1", "E2", "E3"])
    algorithm_comparison_table(results, suites=["E4", "E5"])
    scalability_table(results)

    # Random scenarios summary
    random_results = [r for r in results if r.suite == "E6"]
    if random_results:
        print("\n=== Random Scenarios (E6) ===", file=sys.stderr)
        algorithm_comparison_table(random_results)
