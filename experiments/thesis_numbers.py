"""Compute the numbers that the thesis marks with ``\\rerun{...}``.

The thesis wraps every number, count, time and ratio that the rerun of the
evaluation regenerates in ``\\rerun{...}`` (Chapters 1, 7 and 9, the
abstract and Appendix A).  This script finds every occurrence in a thesis
checkout, computes the new value(s) from the result logs, and writes a
Markdown report that lists, for each occurrence, the file and line, the
current text, the new values, and a proposed replacement.  Where a marked
sentence is qualitative or cannot be computed from the logs, the report says
so and gives the data that bear on it.  The author edits the thesis; this
script never changes it.

Inputs (only ``--results`` is required)::

    python -m experiments.thesis_numbers \\
        --results results/thesis-eval-900.jsonl \\
        --certificates results/thesis-eval-900.cert.jsonl \\
        --repeat results/thesis-eval-300.jsonl \\
        --encoding results/encoding-build.jsonl \\
        --thesis-dir ../../thesis \\
        --output results/thesis-numbers.md [--json results/thesis-numbers.json]

``--repeat`` is the second sweep with a limit of 300 s (Section 7.2), from
which the variation between two runs of the same pair is measured: the
median, over the (instance, configuration) pairs solved in both sweeps, of
``|t1 - t2| / min(t1, t2)``.  ``--encoding`` holds the times to build the
formula F (``python -m experiments.encoding_times``).

Definitions used throughout (they reproduce the numbers of the examined
thesis from the examined log, see ``tests/test_experiments.py``):

* a run is solved if the certifier accepted its equilibrium;
* the speed-up of symmetry reduction on an (instance, algorithm) pair that
  both settings solve is the time without it divided by the time with it;
* "at least twice as fast" means a speed-up of at least 2, "at least twice
  as slow" a speed-up of at most 1/2;
* a median speed-up is "within the measured variation" ``v`` if it lies in
  ``[1/(1+v), 1+v]``.

The report reads logs of both protocols of the evaluation: the first run
(iteration bound ``N = 30`` everywhere, initial play from ``smt(F)``) and the
revised protocol (no bound for the direct runs, initial play from
``smt(F, f)``, CEGAR-NE inner bound 30 at a coarse and none at the discrete
partition).  Besides the markers it reports the protocol of the log, the
share of returned equilibria in which every agent has reward ``m``, the
runs that returned their initial play, the median time of the initial query
per algorithm with and without symmetry reduction (a run whose initial query
did not finish within the limit counts as taking at least the limit), and
the outcomes of the runs with the reason each run stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .results import (
    ALGS,
    FAMILIES,
    FAMILY_LABEL,
    THESIS_MARKER,
    Results,
    Run,
    config_label,
    is_thesis_checkout,
    load,
)

# ---------------------------------------------------------------------------
# Finding the markers
# ---------------------------------------------------------------------------

SKIP_DIRS = {"build", "examination", "web", ".git"}


@dataclass(frozen=True)
class Occurrence:
    """One ``\\rerun{...}`` in a thesis source file."""

    path: str  # relative to the thesis directory
    line: int  # 1-based line of the opening brace
    index: int  # 1-based position among the markers of that line
    count: int  # number of markers on that line
    content: str
    prefix: str  # text of the line before the marker

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"

    @property
    def text(self) -> str:
        """The content with runs of whitespace collapsed."""
        return " ".join(self.content.split())


def _strip_comment(line: str) -> str:
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def find_reruns(thesis_dir: str | Path) -> list[Occurrence]:
    """Every ``\\rerun{...}`` outside comments, in file order.

    The argument may span lines and contain nested braces.  The macro
    definition in ``thesis.tex`` (``\\newcommand{\\rerun}``) is not a use.
    """
    root = Path(thesis_dir)
    found: list[Occurrence] = []
    files = sorted(
        p
        for p in root.rglob("*.tex")
        if not any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1])
    )
    for path in files:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        code_lines = [_strip_comment(line) for line in raw_lines]
        text = "\n".join(code_lines)
        starts = [m.end() for m in re.finditer(r"\\rerun\{", text)]
        per_line: dict[int, list[Occurrence]] = {}
        for start in starts:
            depth = 1
            i = start
            while i < len(text) and depth:
                c = text[i]
                if c == "\\":
                    i += 2
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1
            content = text[start : i - 1]
            line_no = text.count("\n", 0, start) + 1
            line_start = text.rfind("\n", 0, start) + 1
            prefix = text[line_start : start - len("\\rerun{")]
            occ = Occurrence(
                path=path.relative_to(root).as_posix(),
                line=line_no,
                index=0,
                count=0,
                content=content,
                prefix=prefix,
            )
            per_line.setdefault(line_no, []).append(occ)
        for line_no in sorted(per_line):
            occs = per_line[line_no]
            for k, occ in enumerate(occs, 1):
                found.append(
                    Occurrence(
                        occ.path, occ.line, k, len(occs), occ.content, occ.prefix
                    )
                )
    return found


def find_red_groups(thesis_dir: str | Path) -> list[tuple[str, int, str]]:
    """``\\color{red}`` outside ``\\rerun`` in the experiment chapter and appendix."""
    out = []
    root = Path(thesis_dir)
    for sub in ("chapters/experiment", "appendices/experiment"):
        for path in sorted((root / sub).glob("*.tex")):
            for k, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                code = _strip_comment(line)
                if "\\color{red}" in code:
                    out.append((path.relative_to(root).as_posix(), k, line.strip()))
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]

PROSE = {
    "ibis": r"\gls{ibis}",
    "sibis": r"\gls{sibis}",
    "qibis": r"\gls{qibis}",
    "cegar-sibis": "CEGAR-NE",
}


def words(n: int) -> str:
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def t1(x: float | None) -> str:
    return "--" if x is None else f"{x:.1f}"


def r2(x: float | None) -> str:
    return "--" if x is None else f"{x:.2f}"


def pct(x: float | None) -> str:
    return "--" if x is None else f"{round(100 * x)}\\%"


def and_list(items: list[str], sep: str = " and~") -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + sep + items[-1]


def fraction_phrase(x: float | None) -> str:
    """``0.33`` -> ``a third``; used for 'a third of', 'a quarter as many'."""
    if x is None:
        return "--"
    table = [
        (0.5, "half"),
        (1 / 3, "a third"),
        (0.25, "a quarter"),
        (0.2, "a fifth"),
        (2 / 3, "two thirds"),
        (0.75, "three quarters"),
    ]
    for value, phrase in table:
        if abs(x - value) <= 0.035:
            return phrase
    return f"{x:.2f} times"


def inst_label(r: Run) -> str:
    return f"{r.grid} (n={r.agents})"


def inst_prose(r: Run) -> str:
    """'the $6{\\times}4$ grid with three agents' style."""
    m = re.fullmatch(r"(\d+)x(\d+)", r.grid)
    if r.family == "symmetric" and m:
        shape = f"the ${m.group(1)}{{\\times}}{m.group(2)}$ grid"
    elif r.family == "asymmetric":
        shape = f"the random territory \\texttt{{{r.grid}}}"
    else:
        shape = f"\\texttt{{{r.grid}}}"
    return f"{shape} with {words(r.agents)} agents"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _orbit_blocks_from_grid(grid_path: str) -> int | None:
    """Blocks of the orbit partition (logs of the examined runner do not record it)."""
    try:
        from rcmas.abstraction import orbit_partition
        from rcmas.model import Territory

        from .provenance import TOOL_ROOT

        path = TOOL_ROOT / grid_path
        with open(path, encoding="utf-8") as f:
            t = Territory.from_ascii(f)
        return len(orbit_partition(t).blocks)
    except Exception:  # noqa: BLE001 - the report states what it could not compute
        return None


REASON_LABEL = {
    "cycle": "recurrence",
    "max_iters": "bound",
    "failure_no_candidate": "no candidate",
    "failure_refuted": "refuted",
    "unknown": "unknown",
    "unsat": "unsatisfiable",
    "timeout": "limit",
    "max_refinements": "refinement cap",
}


def outcome_reason(r: Run) -> str:
    """``"n/c (recurrence)"``-style label of how a run stopped."""
    if r.outcome in ("ne", "t/o", "error"):
        if r.outcome == "ne" and not r.solved:
            return "ne (not certified)"
        return r.outcome
    label = REASON_LABEL.get(str(r.reason), str(r.reason))
    return f"{r.outcome} ({label})"


def bound_text(n: int | None) -> str:
    return "none" if n is None else str(n)


def protocol_text(p: dict[str, Any]) -> str:
    return (
        f"direct runs N = {bound_text(p.get('max_iters'))}, initial play "
        f"{'smt(F, f)' if p.get('initial') == 'collective' else 'smt(F)'} "
        f"({p.get('initial')}), CEGAR-NE inner N = "
        f"{bound_text(p.get('cegar_inner_max_iters'))} at a coarse and "
        f"{bound_text(p.get('cegar_discrete_max_iters'))} at the discrete partition"
    )


class Analysis:
    """The quantities behind the ``\\rerun`` markers, computed from the logs."""

    def __init__(
        self,
        d: Results,
        repeat: Results | None = None,
        certificates: dict[str, int] | None = None,
        encoding: list[dict[str, Any]] | None = None,
    ) -> None:
        self.d = d
        self.repeat = repeat
        self.cert_counts = certificates
        self.encoding = encoding
        self.N = len(d.instances)
        self.limit = d.time_limit_s
        self.algs = [a for a in ALGS if a in d.algorithms]

    # -- counts ---------------------------------------------------------
    def count(self, alg: str, sym: bool) -> int:
        return len(self.d.solved_instances(alg, sym))

    def ranking(
        self, sym: bool, algs: list[str] | None = None
    ) -> list[tuple[str, int]]:
        algs = algs or self.algs
        return sorted(((a, self.count(a, sym)) for a in algs), key=lambda x: -x[1])

    def best(self, sym: bool) -> tuple[list[str], int]:
        rank = self.ranking(sym)
        top = rank[0][1] if rank else 0
        return [a for a, c in rank if c == top], top

    def median(self, alg: str, sym: bool) -> float | None:
        return self.d.median_time(alg, sym)

    def solved_runs(self) -> list[Run]:
        return [r for r in self.d.runs if r.solved]

    def unsolved_runs(self) -> list[Run]:
        return [r for r in self.d.runs if not r.solved]

    def solved_by_any(self, inst: tuple[str, int]) -> list[tuple[str, bool]]:
        return [
            (a, s)
            for a, s in self.d.configs
            if (r := self.d.get(inst, a, s)) is not None and r.solved
        ]

    # -- symmetry -------------------------------------------------------
    def speedups(self) -> list[tuple[float, Run, Run]]:
        out = []
        for inst in self.d.instances:
            for alg in self.algs:
                r0, r1 = self.d.get(inst, alg, False), self.d.get(inst, alg, True)
                if r0 and r1 and r0.solved and r1.solved and r1.time_s > 0:
                    out.append((r0.time_s / r1.time_s, r0, r1))
        return out

    def median_speedup(self, family: str | None = None) -> float | None:
        xs = [
            s for s, r0, _ in self.speedups() if family is None or r0.family == family
        ]
        return statistics.median(xs) if xs else None

    def only(self, with_symmetry: bool) -> list[Run]:
        out = []
        for inst in self.d.instances:
            for alg in self.algs:
                r_on, r_off = self.d.get(inst, alg, True), self.d.get(inst, alg, False)
                a, b = (r_on, r_off) if with_symmetry else (r_off, r_on)
                if a and a.solved and not (b and b.solved):
                    out.append(a)
        return out

    def variation(self) -> tuple[float | None, int]:
        """Median relative difference of two runs of a pair, and the number of pairs."""
        if self.repeat is None:
            return None, 0
        xs = []
        for r in self.d.runs:
            r2_ = self.repeat.get(r.instance, r.algorithm, r.symmetry)
            if r.solved and r2_ is not None and r2_.solved:
                lo = min(r.time_s, r2_.time_s)
                if lo > 0:
                    xs.append(abs(r.time_s - r2_.time_s) / lo)
        return (statistics.median(xs) if xs else None), len(xs)

    def within_variation(self, x: float | None) -> bool | None:
        v, _ = self.variation()
        if x is None or v is None:
            return None
        return 1 / (1 + v) <= x <= 1 + v

    # -- protocol and initial play ---------------------------------------
    @property
    def protocol(self) -> dict[str, Any]:
        return self.d.protocol

    @property
    def max_iters(self) -> int | None:
        return self.d.max_iters

    def all_m_share(self, runs: list[Run] | None = None) -> tuple[int, int]:
        """(equilibria in which every agent has reward ``m``, equilibria) among solved runs."""
        solved = self.solved_runs() if runs is None else [r for r in runs if r.solved]
        return sum(r.all_at_bound for r in solved), len(solved)

    def returned_initial(self, alg: str, sym: bool) -> list[Run] | None:
        """Solved runs of a configuration that returned their initial play (``None``: not recorded)."""
        runs = self.d.runs_of(alg, sym)
        if not any(r.has_initial_fields for r in runs):
            return None
        return [r for r in runs if r.solved and r.initial_was_ne]

    def initial_all_m(self, alg: str, sym: bool) -> tuple[int, int] | None:
        """(runs whose initial play gives every agent ``m``, runs with a recorded initial play)."""
        runs = [r for r in self.d.runs_of(alg, sym) if r.initial_rewards is not None]
        if not any(r.has_initial_fields for r in self.d.runs_of(alg, sym)):
            return None
        return sum(bool(r.initial_all_m) for r in runs), len(runs)

    def initial_query_times(
        self, alg: str, sym: bool
    ) -> tuple[float | None, int, int] | None:
        """Median time of the initial query, runs counted, and runs whose query did not finish.

        A run whose initial query was not answered (cut off at the limit,
        or answered unknown at the Z3 timeout) counts as taking at least the
        limit (``inf`` in the median).  ``None`` if the log does not record
        the initial query.
        """
        runs = [r for r in self.d.runs_of(alg, sym) if r.has_initial_fields]
        if not runs:
            return None
        xs: list[float] = []
        unfinished = 0
        for r in runs:
            t = r.initial_query_s
            if t is not None and r.initial_query_finished:
                xs.append(t)
            elif r.outcome in ("t/o", "unk"):
                xs.append(float("inf"))
                unfinished += 1
        med = statistics.median(xs) if xs else None
        return med, len(xs), unfinished

    def outcomes_by_reason(self) -> dict[tuple[str, bool], dict[str, int]]:
        """For every configuration: ``"outcome (reason)" -> number of runs``."""
        out: dict[tuple[str, bool], dict[str, int]] = {}
        for cfg in self.d.configs:
            counts: dict[str, int] = {}
            for r in self.d.runs_of(*cfg):
                key = outcome_reason(r)
                counts[key] = counts.get(key, 0) + 1
            out[cfg] = counts
        return out

    # -- time curves ----------------------------------------------------
    def solved_within(self, alg: str, sym: bool, t: float) -> int:
        return sum(1 for r in self.d.solved_of(alg, sym) if r.time_s <= t)

    def order_violations(self, sym: bool) -> list[tuple[float, list[tuple[str, int]]]]:
        rank = [a for a, _ in self.ranking(sym)]
        times = sorted({r.time_s for a in rank for r in self.d.solved_of(a, sym)})
        bad = []
        for t in times:
            counts = [(a, self.solved_within(a, sym, t)) for a in rank]
            if any(counts[i][1] < counts[i + 1][1] for i in range(len(counts) - 1)):
                bad.append((t, counts))
        return bad

    # -- CEGAR ----------------------------------------------------------
    def orbit_blocks(self, inst: tuple[str, int]) -> int | None:
        """``|T/H|`` of the instance's territory, from the log or from the grid file."""
        blocks = {
            r.orbit_blocks
            for r in self.d.runs
            if r.instance == inst and r.orbit_blocks is not None
        }
        if blocks:
            return min(blocks)
        return _orbit_blocks_from_grid(inst[0])

    def discrete_orbit_instances(self) -> list[tuple[str, int]]:
        out = []
        for inst in self.d.instances:
            blocks = self.orbit_blocks(inst)
            if blocks is not None and blocks == self.d.instance_info(inst).sectors:
                out.append(inst)
        return out

    def cegar_ratio(self, insts: list[tuple[str, int]]) -> tuple[float | None, int]:
        xs = []
        for inst in insts:
            for sym in (False, True):
                rs, rc = self.d.get(inst, "sibis", sym), self.d.get(
                    inst, "cegar-sibis", sym
                )
                if rs and rc and rs.solved and rc.solved and rs.time_s > 0:
                    xs.append(rc.time_s / rs.time_s)
        return (statistics.median(xs) if xs else None), len(xs)

    def set_diff(self, a: str, b: str, sym: bool) -> list[Run]:
        """Runs of *a* that solve an instance that *b* does not solve (same setting)."""
        sb = self.d.solved_instances(b, sym)
        return [r for r in self.d.solved_of(a, sym) if r.instance not in sb]

    # -- size -----------------------------------------------------------
    def size_profile(self) -> dict[str, Any]:
        insts = self.d.instances
        sizes = sorted({self.d.instance_info(i).sectors for i in insts})
        unsolved = [i for i in insts if not self.solved_by_any(i)]
        un_sizes = sorted({self.d.instance_info(i).sectors for i in unsolved})
        first_unsolved = un_sizes[0] if un_sizes else None
        all_up_to = max(
            (s for s in sizes if first_unsolved is None or s < first_unsolved),
            default=None,
        )
        biggest = sizes[-1] if sizes else None
        big_insts = [i for i in insts if self.d.instance_info(i).sectors == biggest]
        big_solved = []
        for i in big_insts:
            runs = [self.d.get(i, a, s) for a, s in self.solved_by_any(i)]
            big_solved.append((self.d.instance_info(i), [r for r in runs if r]))
        return {
            "sizes": sizes,
            "all_solved_up_to": all_up_to,
            "first_unsolved_size": first_unsolved,
            "unsolved_at_first": [
                self.d.instance_info(i)
                for i in unsolved
                if self.d.instance_info(i).sectors == first_unsolved
            ],
            "unsolved": [self.d.instance_info(i) for i in unsolved],
            "largest_size": biggest,
            "largest": big_solved,
        }

    def largest_solved_any(self) -> Run | None:
        best = None
        for r in self.solved_runs():
            if best is None or (r.sectors, r.agents) > (best.sectors, best.agents):
                best = r
        return best

    def agents_solved_at(self, sectors: int) -> list[int]:
        return sorted({r.agents for r in self.solved_runs() if r.sectors == sectors})

    def territories_with_unsolved(self) -> list[tuple[str, list[int], list[int]]]:
        by_grid: dict[str, tuple[list[int], list[int]]] = {}
        for inst in self.d.instances:
            info = self.d.instance_info(inst)
            solved, unsolved = by_grid.setdefault(info.grid, ([], []))
            (solved if self.solved_by_any(inst) else unsolved).append(info.agents)
        return [(g, s, u) for g, (s, u) in by_grid.items() if u]

    # -- certifier ------------------------------------------------------
    def certificate_counts(self) -> dict[str, int]:
        if self.cert_counts is not None:
            return dict(self.cert_counts)
        counts = {"accepted": 0, "rejected": 0, "unchecked": 0}
        for r in self.d.runs:
            if r.found_ne:
                counts[r.certificate or "unchecked"] = (
                    counts.get(r.certificate or "unchecked", 0) + 1
                )
        return counts

    # -- provenance -----------------------------------------------------
    def sessions(self) -> list[dict[str, Any]]:
        return self.d.sessions

    def provenance_values(self, key: Callable[[dict[str, Any]], Any]) -> list[Any]:
        vals = []
        for s in self.sessions():
            v = key(s)
            if v not in vals:
                vals.append(v)
        return vals


# ---------------------------------------------------------------------------
# Rules: one per kind of marker
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    values: list[tuple[str, str]] = field(default_factory=list)
    proposed: str | None = None
    note: str | None = None
    computable: bool = True


Rule = tuple[
    str, Callable[[Occurrence], bool], Callable[[Analysis, Occurrence], Finding]
]


def _contains(*parts: str) -> Callable[[Occurrence], bool]:
    def match(o: Occurrence) -> bool:
        return all(p in o.text for p in parts)

    return match


def _short(content: str, prefix_end: str) -> Callable[[Occurrence], bool]:
    def match(o: Occurrence) -> bool:
        return o.text == content and o.prefix.rstrip().endswith(prefix_end.rstrip())

    return match


def _count_rule(alg: str, sym: bool) -> Callable[[Analysis, Occurrence], Finding]:
    def f(a: Analysis, o: Occurrence) -> Finding:
        c = a.count(alg, sym)
        rank = a.ranking(sym)
        return Finding(
            values=[
                (f"{config_label(alg, sym)} instances solved", f"{c} of {a.N}"),
                (
                    "ranking " + ("with" if sym else "without") + " symmetry",
                    ", ".join(f"{config_label(x, sym)} {n}" for x, n in rank),
                ),
            ],
            proposed=str(c),
        )

    return f


def _counts_on(a: Analysis, o: Occurrence) -> Finding:
    order = ["sibis", "cegar-sibis", "ibis", "qibis"]
    nums = [str(a.count(x, True)) for x in order if x in a.algs]
    rank_off = [x for x, _ in a.ranking(False)]
    rank_on = [x for x, _ in a.ranking(True)]
    note = None
    if rank_off != rank_on or rank_off != order:
        note = (
            "The sentence lists the algorithms as S-IBIS, CEGAR-NE, IBIS, Q-IBIS, in "
            "decreasing order of solved instances. New order without symmetry: "
            + ", ".join(config_label(x, False) for x in rank_off)
            + "; with symmetry: "
            + ", ".join(config_label(x, True) for x in rank_on)
            + ". Reorder the sentence if needed."
        )
    return Finding(
        values=[("with symmetry (S-IBIS, CEGAR-NE, IBIS, Q-IBIS)", ", ".join(nums))],
        proposed=and_list(nums),
        note=note,
    )


def _cactus_order(a: Analysis, o: Occurrence) -> Finding:
    vals = []
    holds = True
    for sym in (False, True):
        bad = a.order_violations(sym)
        rank = ", ".join(config_label(x, sym) for x, _ in a.ranking(sym))
        if bad:
            holds = False
            t, counts = bad[0]
            vals.append(
                (
                    f"order {'with' if sym else 'without'} symmetry ({rank})",
                    f"violated at {len(bad)} times, first at {t:.1f} s: "
                    + ", ".join(f"{config_label(x, sym)} {c}" for x, c in counts),
                )
            )
        else:
            vals.append(
                (
                    f"order {'with' if sym else 'without'} symmetry ({rank})",
                    "kept at every time",
                )
            )
    return Finding(
        values=vals,
        proposed=o.text if holds else None,
        note=(
            None if holds else "The claim does not hold; rewrite or drop the sentence."
        ),
    )


def _timeouts_unsolved(a: Analysis, o: Occurrence) -> Finding:
    un = a.unsolved_runs()
    to = [r for r in un if r.outcome == "t/o"]
    share = len(to) / len(un) if un else None
    s_un = [
        i
        for i in a.d.instances
        if not (r := a.d.get(i, "sibis", False)) or not r.solved
    ]
    s_runs = [a.d.get(i, "sibis", False) for i in s_un]
    outcomes: dict[str, int] = {}
    for r in s_runs:
        key = r.outcome if r else "missing"
        outcomes[key] = outcomes.get(key, 0) + 1
    sizes = [r.sectors for r in s_runs if r]
    all_to = all(r is not None and r.outcome == "t/o" for r in s_runs)
    from collections import Counter

    by_outcome = Counter(r.outcome for r in un)
    vals = [
        (
            "unsolved runs, all configurations",
            f"{len(un)}: " + ", ".join(f"{k} {v}" for k, v in by_outcome.most_common()),
        ),
        ("share of timeouts among them", pct(share)),
        ("instances S-IBIS (no symmetry) does not solve", str(len(s_un))),
        (
            "their outcomes",
            ", ".join(f"{k} {v}" for k, v in outcomes.items()) or "none",
        ),
        ("smallest territory among them", f"{min(sizes)} sectors" if sizes else "--"),
    ]
    most = share is not None and share > 0.5
    if s_un:
        tail = (
            f"the {len(s_un)} instances that \\gls{{sibis}} does not solve without symmetry reduction are "
            + (
                "timeouts"
                if all_to
                else "unsolved runs ("
                + ", ".join(f"{k}: {v}" for k, v in outcomes.items())
                + ")"
            )
            + f" on territories with at least {min(sizes)} sectors"
        )
    else:
        tail = "\\gls{sibis} solves every instance without symmetry reduction"
    head = (
        "Timeouts account for most of the unsolved runs"
        if most
        else "Timeouts account for a minority of the unsolved runs"
    )
    return Finding(values=vals, proposed=f"{head}, and {tail}.")


def _asset(a: Analysis, o: Occurrence) -> Finding:
    m = re.search(r"\\input\{([^}]+)\}", o.content)
    target = m.group(1) if m else o.text
    return Finding(
        values=[
            ("regenerated file", f"{target}.tex (python -m experiments.thesis_assets)")
        ],
        proposed=o.text,
        note="Generated asset: once the regenerated file is in place, remove the \\rerun wrapper.",
    )


def _caption_markers(a: Analysis, o: Occurrence) -> Finding:
    present = sorted({r.outcome for r in a.d.runs if not r.solved})
    return Finding(
        values=[("unsolved outcomes in the log", ", ".join(present) or "none")],
        note="This file is regenerated by experiments.thesis_assets; the caption lists exactly the markers that occur.",
        proposed=None,
    )


def _sibis_median(a: Analysis, o: Occurrence) -> Finding:
    ms, mi = a.median("sibis", False), a.median("ibis", False)
    ratio = ms / mi if ms is not None and mi else None
    biggest = a.size_profile()["largest_size"]
    cfgs = sorted(
        {
            config_label(r.algorithm, r.symmetry)
            for r in a.solved_runs()
            if r.sectors == biggest
        }
    )
    algs30 = sorted({r.algorithm for r in a.solved_runs() if r.sectors == biggest})
    only_sibis = set(algs30) <= {"sibis", "cegar-sibis"} and bool(algs30)
    vals = [
        ("median time S-IBIS (no symmetry)", t1(ms)),
        ("median time IBIS (no symmetry)", t1(mi)),
        ("ratio", r2(ratio) + f" ({fraction_phrase(ratio)})"),
        (
            f"configurations that solve a {biggest}-sector instance",
            ", ".join(cfgs) or "none",
        ),
    ]
    if only_sibis:
        tail = f", and only \\gls{{sibis}}, run directly or as the inner synthesiser of CEGAR-NE, solves any of the {biggest}-sector instances"
    elif algs30:
        tail = f", and the {biggest}-sector instances are solved by " + and_list(
            [PROSE.get(x, x) for x in algs30], " and "
        )
    else:
        tail = f", and no configuration solves a {biggest}-sector instance"
    rel = "a smaller" if ratio is not None and ratio < 1 else "not smaller"
    proposed = f"The median time of \\gls{{sibis}} on solved instances, {t1(ms)} seconds, is {fraction_phrase(ratio)} of that of \\gls{{ibis}}{tail}."
    note = (
        None
        if ratio is not None and ratio < 1
        else f"The S-IBIS median is {rel} than the IBIS median; rewrite."
    )
    return Finding(values=vals, proposed=proposed, note=note)


def _curves_differ(a: Analysis, o: Occurrence) -> Finding:
    ts = [1, 10, 60, 300, a.limit]
    vals = []
    for alg in a.algs:
        off = [a.solved_within(alg, False, t) for t in ts]
        on = [a.solved_within(alg, True, t) for t in ts]
        vals.append(
            (
                f"{config_label(alg, False)} solved within "
                + "/".join(f"{t:g}" for t in ts)
                + " s",
                "without "
                + "/".join(map(str, off))
                + "; with "
                + "/".join(map(str, on)),
            )
        )
    return Finding(
        values=vals,
        computable=False,
        note="Qualitative: compare the curves of Figure 7.2; the counts above show where they separate.",
    )


def _well_within(a: Analysis, o: Occurrence) -> Finding:
    solved = a.solved_runs()
    over = [r for r in solved if r.time_s > 300]
    vals = [
        ("solved runs", str(len(solved))),
        ("solved runs above 300 s", str(len(over))),
    ]
    most_fast = True
    for alg, sym in a.d.configs:
        rs = a.d.solved_of(alg, sym)
        if not rs:
            continue
        f60 = sum(r.time_s <= 60 for r in rs) / len(rs)
        f100 = sum(r.time_s <= 100 for r in rs) / len(rs)
        most_fast &= f100 > 0.5
        vals.append(
            (
                f"{config_label(alg, sym)} share solved within 60 s / 100 s",
                f"{pct(f60)} / {pct(f100)}",
            )
        )
    proposed = (
        f"Most runs that return an equilibrium do so well within the limit. Only {len(over)} of the {len(solved)} solved runs took more than 300 seconds"
        + (
            ", and every configuration solves most of its instances within tens of seconds, after which its curve in Figure~\\ref{fig:exp:cactus} flattens."
            if most_fast
            else "."
        )
    )
    note = (
        None
        if most_fast
        else "Some configuration solves at most half of its instances within 100 s; the second clause was dropped."
    )
    return Finding(values=vals, proposed=proposed, note=note)


def _sibis_vs_ibis(a: Analysis, o: Occurrence) -> Finding:
    vals = []
    more = smaller = larger = True
    for sym in (False, True):
        cs, ci = a.count("sibis", sym), a.count("ibis", sym)
        ms, mi = a.median("sibis", sym), a.median("ibis", sym)
        bs, bi = a.d.largest_solved("sibis", sym), a.d.largest_solved("ibis", sym)
        more &= cs > ci
        smaller &= ms is not None and mi is not None and ms < mi
        key = lambda r: (r.sectors, r.agents) if r else (-1, -1)  # noqa: E731
        larger &= key(bs) > key(bi)
        st = "with" if sym else "without"
        vals.append((f"S-IBIS vs IBIS {st} symmetry: solved", f"{cs} vs {ci}"))
        vals.append(
            (f"S-IBIS vs IBIS {st} symmetry: median time", f"{t1(ms)} vs {t1(mi)}")
        )
        vals.append(
            (
                f"S-IBIS vs IBIS {st} symmetry: largest solved",
                f"{inst_label(bs) if bs else '--'} vs {inst_label(bi) if bi else '--'}",
            )
        )
    holds = more and smaller and larger
    note = None
    if not holds:
        failed = [
            n
            for n, ok in (
                ("more instances", more),
                ("smaller median", smaller),
                ("larger largest instance", larger),
            )
            if not ok
        ]
        note = "Does not hold in both settings: " + ", ".join(failed) + ". Rewrite."
    return Finding(values=vals, proposed=o.text if holds else None, note=note)


def _qibis_fewest(a: Analysis, o: Occurrence) -> Finding:
    direct = [x for x in ("ibis", "sibis", "qibis") if x in a.algs]
    fewest = all(
        a.count("qibis", s) <= min(a.count(x, s) for x in direct) for s in (False, True)
    )
    pairs = []
    for i in a.d.instances:
        q, b = a.d.get(i, "qibis", False), a.d.get(i, "ibis", False)
        if q is not None and b is not None and q.solved and b.solved:
            pairs.append((q.time_s, b.time_s))
    both = pairs
    mq = statistics.median(x for x, _ in pairs) if pairs else None
    mi = statistics.median(y for _, y in pairs) if pairs else None
    tab_q, tab_i = a.median("qibis", False), a.median("ibis", False)
    vals = [
        ("Q-IBIS fewest of the direct synthesisers in both settings", str(fewest)),
        ("instances both Q-IBIS and IBIS solve (no symmetry)", str(len(both))),
        ("median Q-IBIS / IBIS on them", f"{t1(mq)} / {t1(mi)}"),
        ("Table 7.1 medians Q-IBIS / IBIS (no symmetry)", f"{t1(tab_q)} / {t1(tab_i)}"),
    ]
    proposed = (
        (
            "\\gls{qibis} solves the fewest instances of the three direct synthesisers at this time limit. "
            if fewest
            else ""
        )
        + f"On the {len(both)} instances that \\gls{{qibis}} and \\gls{{ibis}} both solve without symmetry reduction, the median time of \\gls{{qibis}} is {t1(mq)} seconds against {t1(mi)} seconds for \\gls{{ibis}}"
    )
    if (
        tab_q is not None
        and tab_i is not None
        and tab_q < tab_i
        and mq is not None
        and mi is not None
        and mq > mi
    ):
        proposed += ", so its smaller median in Table~\\ref{tab:exp:summary} comes from the instances that it solves."
        note = None
    else:
        proposed += "."
        note = "The clause about the smaller median in Table 7.1 no longer applies (see the medians above); it was dropped."
    if not fewest:
        note = (
            note or ""
        ) + " Q-IBIS does not solve the fewest instances in both settings."
    return Finding(values=vals, proposed=proposed, note=note)


def _iterations(a: Analysis, o: Occurrence) -> Finding:
    its = [
        r.iterations
        for r in a.solved_runs()
        if r.algorithm in ("ibis", "sibis") and r.iterations is not None
    ]
    mx = max(its) if its else None
    q_nc = [r for r in a.d.runs if r.algorithm == "qibis" and r.outcome == "n/c"]
    q_bound = [r for r in q_nc if r.reason == "max_iters"]
    q_cycle = [r for r in q_nc if r.reason == "cycle"]
    all_nc = [r for r in a.d.runs if r.outcome == "n/c"]
    from collections import Counter

    by_cfg = Counter(config_label(r.algorithm, r.symmetry) for r in all_nc)
    N = a.max_iters
    vals = [
        ("max iterations of a solved IBIS/S-IBIS run", str(mx)),
        (
            "Q-IBIS runs n/c (bound / recurrence)",
            f"{len(q_nc)} ({len(q_bound)} / {len(q_cycle)})",
        ),
        (
            "n/c runs per configuration",
            ", ".join(f"{k} {v}" for k, v in sorted(by_cfg.items())) or "none",
        ),
    ]
    head = f"\\gls{{ibis}} and \\gls{{sibis}} need at most {words(mx) if mx is not None else '--'} improvement iterations on every instance they solve, "
    if N is None:
        proposed = (
            head
            + f"while {len(q_cycle)} \\gls{{qibis}} runs return to an earlier play without an equilibrium (``n/c'')."
        )
    else:
        proposed = (
            head
            + f"while {len(q_bound)} \\gls{{qibis}} runs reach the iteration bound of {N} without an equilibrium (``n/c'')"
            + (f" and {len(q_cycle)} return to an earlier play" if q_cycle else "")
            + "."
        )
    note = (
        "The direct runs have no iteration bound in this protocol, so a Q-IBIS run ends n/c only by recurrence."
        if N is None
        else None
    )
    others = sum(v for k, v in by_cfg.items() if not k.startswith("Q-IBIS"))
    if others:
        note = (
            note + " " if note else ""
        ) + f"{others} runs of other configurations also end n/c (see above)."
    return Finding(values=vals, proposed=proposed, note=note)


def _speedup_main(a: Analysis, o: Occurrence) -> Finding:
    sp = a.speedups()
    med = a.median_speedup()
    v, nv = a.variation()
    within = a.within_variation(med)
    vals = [
        ("pairs solved with and without symmetry", str(len(sp))),
        ("median speed-up", r2(med)),
        (
            "measured variation",
            f"{pct(v)} over {nv} pairs" if v is not None else "no repeat sweep given",
        ),
        ("within the variation", str(within)),
    ]
    phrase = (
        "outside the measured variation"
        if within is False
        else "within the measured variation"
    )
    note = (
        None
        if within is not None
        else "Needs the repeat sweep (--repeat) to decide 'within the measured variation'."
    )
    if within is False:
        note = "The median speed-up lies outside the measured variation; revise the interpretation."
    return Finding(
        values=vals,
        proposed=f"Over the {len(sp)} (instance, algorithm) pairs solved both with and without symmetry reduction, the median speed-up is {r2(med)}, {phrase} (Figure~\\ref{{fig:exp:scatter}}(a)).",
        note=note,
    )


def _twice(a: Analysis, o: Occurrence) -> Finding:
    sp = a.speedups()
    fast = sum(1 for s, _, _ in sp if s >= 2)
    slow = sum(1 for s, _, _ in sp if s <= 0.5)
    return Finding(
        values=[
            ("pairs at least twice as fast with symmetry", str(fast)),
            ("pairs at least twice as slow", str(slow)),
        ],
        proposed=f"In {fast} pairs the run with symmetry reduction is at least twice as fast, and in {slow} pairs it is at least twice as slow.",
    )


def _only(a: Analysis, o: Occurrence) -> Finding:
    on, off = a.only(True), a.only(False)
    return Finding(
        values=[
            (
                "runs solved only with symmetry",
                f"{len(on)}: "
                + ", ".join(
                    f"{config_label(r.algorithm, True)} {inst_label(r)}" for r in on
                ),
            ),
            (
                "runs solved only without symmetry",
                f"{len(off)}: "
                + ", ".join(
                    f"{config_label(r.algorithm, False)} {inst_label(r)}" for r in off
                ),
            ),
        ],
        proposed=f"Symmetry reduction solves {len(on)} runs that are unsolved without it, and {len(off)} runs are solved only without it.",
    )


def _per_family(a: Analysis, o: Occurrence) -> Finding:
    meds = {f: a.median_speedup(f) for f in FAMILIES}
    counts = {
        f: sum(1 for _, r0, _ in a.speedups() if r0.family == f) for f in FAMILIES
    }
    sp = a.speedups()
    vals = [
        (f"median speed-up, {FAMILY_LABEL[f]} ({counts[f]} pairs)", r2(meds[f]))
        for f in FAMILIES
    ]
    if sp:
        s, r0, _ = max(sp, key=lambda x: x[0])
        vals.append(
            (
                "largest speed-up",
                f"{r2(s)}: {config_label(r0.algorithm, False)} on {inst_label(r0)}",
            )
        )
        largest = f"the largest speed-up, {r2(s)}, is that of {PROSE.get(r0.algorithm, r0.algorithm)} on {inst_prose(r0)}"
    else:
        largest = "no pair is solved in both settings"
    return Finding(
        values=vals,
        proposed=(
            f"The median speed-up is {r2(meds['symmetric'])} on the square and rectangular grids, {r2(meds['local_symmetry'])} on the L-shaped territories, "
            f"{r2(meds['disconnected'])} on the disconnected territories and {r2(meds['asymmetric'])} on the random territories, and {largest}."
        ),
    )


def _discrete_orbit(a: Analysis, o: Occurrence) -> Finding:
    insts = a.discrete_orbit_instances()
    med, n = a.cegar_ratio(insts)
    names = sorted({a.d.instance_info(i).grid for i in insts})
    return Finding(
        values=[
            (
                "instances with a discrete orbit partition",
                f"{len(insts)} (territories: {', '.join(names) or 'none'})",
            ),
            ("median CEGAR/S-IBIS time ratio on them", f"{r2(med)} over {n} pairs"),
        ],
        proposed=f"This is the case on {len(insts)} of the {a.N} instances, and on them the median ratio of the time of CEGAR-NE to the time of direct \\gls{{sibis}} is {r2(med)} (Figure~\\ref{{fig:exp:scatter}}(b)).",
    )


def _cegar_vs_sibis(sym: bool) -> Callable[[Analysis, Occurrence], Finding]:
    def f(a: Analysis, o: Occurrence) -> Finding:
        cc, cs = a.count("cegar-sibis", sym), a.count("sibis", sym)
        c_only = a.set_diff("cegar-sibis", "sibis", sym)
        s_only = a.set_diff("sibis", "cegar-sibis", sym)
        st = "With" if sym else "Without"
        vals = [
            (f"{st} symmetry: CEGAR / S-IBIS solved", f"{cc} / {cs}"),
            ("CEGAR only", ", ".join(inst_label(r) for r in c_only) or "none"),
            ("S-IBIS only", ", ".join(inst_label(r) for r in s_only) or "none"),
        ]

        def solves(k: int, who: str, other: str, runs: list[Run]) -> str:
            if k == 0:
                return f"{who} solves no instance that {other} does not solve"
            listed = (
                ", " + and_list([inst_prose(r) for r in runs], " and ") + ","
                if k <= 2
                else ""
            )
            return f"{who} solves {k} instance{'s' if k != 1 else ''} that {other} does not{listed}"

        proposed = (
            f"{st} symmetry reduction, CEGAR-NE solves {cc} instances and \\gls{{sibis}} {cs}. "
            + solves(len(c_only), "CEGAR-NE", "\\gls{sibis}", c_only)
            + ", and "
            + solves(len(s_only), "\\gls{sibis}", "CEGAR-NE", s_only)
            + "."
        )
        return Finding(values=vals, proposed=proposed.replace(",,", ","))

    return f


def _sibis_only_discrete(a: Analysis, o: Occurrence) -> Finding:
    disc = set(a.discrete_orbit_instances())
    union = {
        r.instance for s in (False, True) for r in a.set_diff("sibis", "cegar-sibis", s)
    }
    k = sum(1 for i in union if i in disc)
    vals = [
        ("instances only S-IBIS solves (either setting)", str(len(union))),
        ("of them with a discrete orbit partition", str(k)),
    ]
    for s in (False, True):
        part = {r.instance for r in a.set_diff("sibis", "cegar-sibis", s)}
        vals.append(
            (
                f"{'with' if s else 'without'} symmetry: only S-IBIS / discrete",
                f"{len(part)} / {sum(1 for i in part if i in disc)}",
            )
        )
    return Finding(
        values=vals,
        proposed=f"Of the instances that only \\gls{{sibis}} solves, {k} have a discrete orbit partition, so on them the two configurations differ only by the verification queries and by the variation between runs.",
    )


def _cegar_verified(a: Analysis, o: Occurrence) -> Finding:
    ce = [r for r in a.d.runs if r.algorithm == "cegar-sibis"]
    solved = [r for r in ce if r.solved]
    coarse = [r for r in solved if r.cegar_at_discrete is False]
    disc = [r for r in solved if r.cegar_at_discrete]
    fails = [r for r in ce if r.outcome == "fail"]
    ratio = min(
        (r.cegar_final_blocks / r.sectors for r in coarse if r.cegar_final_blocks),
        default=None,
    )
    vals = [
        ("CEGAR equilibria verified at a coarser partition", str(len(coarse))),
        (
            "smallest blocks/sectors among them",
            r2(ratio) + f" ({fraction_phrase(ratio)})" if ratio else "--",
        ),
        ("CEGAR equilibria verified at the discrete partition", str(len(disc))),
        ("CEGAR runs ending with Failure", str(len(fails))),
    ]
    for s in (False, True):
        cs = [r for r in coarse if r.symmetry == s]
        ds = [r for r in disc if r.symmetry == s]
        fs = [r for r in fails if r.symmetry == s]
        vals.append(
            (
                f"{'with' if s else 'without'} symmetry: coarse / discrete / Failure",
                f"{len(cs)} / {len(ds)} / {len(fs)}",
            )
        )
    coarse_part = (
        f"CEGAR-NE verified {len(coarse)} of its equilibria at a partition coarser than the discrete one"
        + (
            f", with as few as {fraction_phrase(ratio)} as many blocks as the territory has sectors"
            if ratio
            else ""
        )
    )
    return Finding(
        values=vals,
        proposed=f"{coarse_part}, and {len(disc)} at the discrete partition, and {len(fails)} runs ended with \\textsc{{Failure}}.",
    )


def _size_main(a: Analysis, o: Occurrence) -> Finding:
    p = a.size_profile()
    vals = [
        ("every instance solved up to", f"{p['all_solved_up_to']} sectors"),
        ("first size with an unsolved instance", str(p["first_unsolved_size"])),
        (
            "unsolved at that size",
            ", ".join(inst_label(r) for r in p["unsolved_at_first"]) or "none",
        ),
    ]
    big = p["largest_size"]
    solved_big = [(info, runs) for info, runs in p["largest"] if runs]
    for info, runs in p["largest"]:
        vals.append(
            (
                f"{big}-sector {inst_label(info)}",
                ", ".join(
                    f"{config_label(r.algorithm, r.symmetry)} {t1(r.time_s)} s"
                    for r in runs
                )
                or "unsolved",
            )
        )
    agents = sorted({info.agents for info, _ in solved_big})
    algs = sorted({r.algorithm for _, runs in solved_big for r in runs})
    times = [r.time_s for _, runs in solved_big for r in runs]
    if p["first_unsolved_size"] is None:
        first = "Every instance is solved by some configuration."
    else:
        un = p["unsolved_at_first"]
        first = (
            f"Every instance with at most {p['all_solved_up_to']} sectors is solved by some configuration. "
            f"At {p['first_unsolved_size']} sectors the first instances remain unsolved, "
            + and_list([inst_prose(r) for r in un], " and ")
        )
    if solved_big:
        by = and_list([PROSE.get(x, x) for x in algs], " and ")
        if set(algs) <= {"sibis", "cegar-sibis"}:
            by = "\\gls{sibis}, directly or inside CEGAR-NE" if len(algs) == 2 else by
        last = (
            f", and of the {words(len(p['largest']))} instances with {big} sectors only those with "
            + and_list([words(x) for x in agents], " and ")
            + f" agents are solved, by {by}, in {min(times):.0f} to {max(times):.0f} seconds."
        )
    else:
        last = f", and none of the {words(len(p['largest']))} instances with {big} sectors is solved."
    return Finding(values=vals, proposed=first + last)


def _both_ends(a: Analysis, o: Occurrence) -> Finding:
    rows = a.territories_with_unsolved()
    vals = [(f"{g}: solved n / unsolved n", f"{s or '-'} / {u}") for g, s, u in rows]
    small_n = [g for g, s, u in rows if s and u and min(u) < min(s)]
    large_n = [g for g, s, u in rows if s and u and max(u) > max(s)]
    vals.append(
        (
            "territories unsolved for fewer agents than some solved n",
            ", ".join(small_n) or "none",
        )
    )
    vals.append(
        (
            "territories unsolved for more agents than some solved n",
            ", ".join(large_n) or "none",
        )
    )
    if small_n and large_n:
        note = "Partly qualitative: check the examples named in the sentence against the rows above."
    else:
        note = "Unsolved instances do not occur at both ends on this benchmark; rewrite the sentence."
    return Finding(values=vals, proposed=None, note=note, computable=False)


def _certifier(a: Analysis, o: Occurrence) -> Finding:
    c = a.certificate_counts()
    total = sum(c.values())
    src = (
        "re-certification file"
        if a.cert_counts is not None
        else "certificates in the log"
    )
    vals = [(f"certifier ({src})", ", ".join(f"{k} {v}" for k, v in c.items()))]
    if c.get("rejected", 0) or c.get("unchecked", 0):
        return Finding(
            values=vals,
            proposed=None,
            note="The certifier did not accept every equilibrium. Investigate before writing this sentence.",
        )
    return Finding(values=vals, proposed=f"It accepted all {total} equilibria.")


def _at_bound(a: Analysis, o: Occurrence) -> Finding:
    solved = a.solved_runs()
    k = sum(r.all_at_bound for r in solved)
    insts = {r.instance for r in solved}
    all_bound_insts = [
        i for i in insts if all(r.all_at_bound for r in solved if r.instance == i)
    ]
    return Finding(
        values=[
            ("solved runs with every reward = m", f"{k} of {len(solved)}"),
            (
                "solved instances whose every returned equilibrium is at the bound",
                f"{len(all_bound_insts)} of {len(insts)}",
            ),
        ],
        proposed=f"and in {k} of the {len(solved)} runs that returned an equilibrium every agent has reward~$m$",
    )


def _summary_sibis_most(a: Analysis, o: Occurrence) -> Finding:
    so, sn = a.count("sibis", False), a.count("sibis", True)
    bo, bn = a.best(False), a.best(True)
    ms, mi = a.median("sibis", False), a.median("ibis", False)
    ratio = ms / mi if ms is not None and mi else None
    q_fewest = all(
        a.count("qibis", s) == min(a.count(x, s) for x in a.algs) for s in (False, True)
    )
    vals = [
        (
            "most solved without symmetry",
            f"{', '.join(config_label(x, False) for x in bo[0])} ({bo[1]})",
        ),
        (
            "most solved with symmetry",
            f"{', '.join(config_label(x, True) for x in bn[0])} ({bn[1]})",
        ),
        ("S-IBIS / IBIS median (no symmetry)", f"{t1(ms)} / {t1(mi)} = {r2(ratio)}"),
        ("Q-IBIS fewest in both settings", str(q_fewest)),
    ]
    note = None
    if "sibis" not in bo[0] or "sibis" not in bn[0]:
        note = "S-IBIS no longer solves the most instances in both settings; rewrite."
    if "against" in o.text:
        io, in_ = a.count("ibis", False), a.count("ibis", True)
        qo, qn = a.count("qibis", False), a.count("qibis", True)
        proposed = (
            f"\\gls{{sibis}} solved the most instances, {so} without and {sn} with symmetry reduction, against {io} and {in_} for \\gls{{ibis}} "
            f"and {qo} and {qn} for \\gls{{qibis}}, and its median time was {fraction_phrase(ratio)} of that of \\gls{{ibis}}."
        )
    else:
        proposed = (
            f"\\gls{{sibis}} solved the most instances, {so} without and {sn} with symmetry reduction, with a median time of {fraction_phrase(ratio)} of that of \\gls{{ibis}}"
            + (", and \\gls{qibis} solved the fewest." if q_fewest else ".")
        )
    return Finding(values=vals, proposed=proposed, note=note)


def _summary_speedup(a: Analysis, o: Occurrence) -> Finding:
    med = a.median_speedup()
    within = a.within_variation(med)
    on, off = a.only(True), a.only(False)
    lies = (
        "is outside the measured variation"
        if within is False
        else "is within the measured variation"
    )
    note = (
        None
        if within is not None
        else "Needs the repeat sweep (--repeat) to decide 'within the measured variation'."
    )
    return Finding(
        values=[
            ("median speed-up", r2(med)),
            ("within the variation", str(within)),
            ("only with / only without", f"{len(on)} / {len(off)}"),
        ],
        proposed=f"its median speed-up of {r2(med)} {lies}, with {len(on)} runs solved only with it and {len(off)} only without it.",
        note=note,
    )


def _generic_speedup(a: Analysis, o: Occurrence) -> Finding:
    """Abstract / Chapter 1 / Chapter 9 sentence on the median speed-up."""
    med = a.median_speedup()
    within = a.within_variation(med)
    on, off = a.only(True), a.only(False)
    verb = "was" if " was " in o.text else "is"
    lies = "lies outside" if within is False else "lies within"
    text = (
        f"Over the (instance, algorithm) pairs solved both with and without symmetry reduction, the median speed-up from symmetry reduction {verb} {r2(med)}, "
        f"which {lies} the variation between runs"
    )
    if "only with symmetry reduction" in o.text:
        text += f", and {len(on)} runs were solved only with symmetry reduction and {len(off)} only without it"
    text += "."
    note = (
        None
        if within is not None
        else "Needs the repeat sweep (--repeat) to decide 'within the variation'."
    )
    return Finding(
        values=[("median speed-up", r2(med)), ("within the variation", str(within))],
        proposed=text,
        note=note,
    )


def _abstract(a: Analysis, o: Occurrence) -> Finding:
    best, n_best = a.best(True)
    cn = a.count("cegar-sibis", True)
    g = _generic_speedup(a, o)
    who = and_list([PROSE.get(x, x) for x in best], " and ")
    text = (
        f"On {a.N} instances with a limit of {a.limit:g} seconds per run and with symmetry reduction, {who} solves the most instances, {n_best}, and CEGAR-NE solves {cn}. "
        + (g.proposed or "")
    )
    return Finding(
        values=[
            ("most solved with symmetry", f"{who} ({n_best})"),
            ("CEGAR-NE with symmetry", str(cn)),
        ]
        + g.values,
        proposed=text,
        note=g.note,
    )


def _cegar_fewer(a: Analysis, o: Occurrence) -> Finding:
    fewer = all(a.count("cegar-sibis", s) < a.count("sibis", s) for s in (False, True))
    c_only = {
        r.instance for s in (False, True) for r in a.set_diff("cegar-sibis", "sibis", s)
    }
    vals = [
        ("CEGAR fewer than S-IBIS in both settings", str(fewer)),
        (
            "instances CEGAR solves and S-IBIS does not (same setting, either setting)",
            str(len(c_only)),
        ),
    ]
    if "51 instances without" in o.text or "instances without and" in o.text:
        co, cn = a.count("cegar-sibis", False), a.count("cegar-sibis", True)
        rel = (
            "fewer than \\gls{sibis} in both settings"
            if fewer
            else "not fewer than \\gls{sibis} in both settings"
        )
        return Finding(
            values=vals,
            proposed=f"CEGAR-NE solved {co} instances without and {cn} with symmetry reduction, {rel}.",
        )
    k = len(c_only)
    tail = (
        "and no instance that \\gls{sibis} did not solve"
        if k == 0
        else f"and {words(k)} instance{'s' if k != 1 else ''} that \\gls{{sibis}} did not solve"
    )
    head = (
        "CEGAR-NE solved fewer instances than \\gls{sibis} in both settings"
        if fewer
        else "CEGAR-NE did not solve fewer instances than \\gls{sibis} in both settings"
    )
    return Finding(values=vals, proposed=f"{head}, {tail}.")


def _size_summary(a: Analysis, o: Occurrence) -> Finding:
    p = a.size_profile()
    big = p["largest_size"]
    solved_big = [info for info, runs in p["largest"] if runs]
    agents = sorted({info.agents for info in solved_big})
    up_to = p["all_solved_up_to"]
    verb = "was" if " was " in o.text else "is"
    first = (
        f"Every instance with at most {up_to} sectors {verb} solved by some configuration"
        if p["first_unsolved_size"] is not None
        else f"Every instance {verb} solved by some configuration"
    )
    if solved_big:
        sv = "were" if verb == "was" else "are"
        last = (
            f", and of the instances with {big} sectors only those with "
            + and_list([words(x) for x in agents], " and ")
            + f" agents {sv} solved"
        )
    else:
        last = f", and no instance with {big} sectors {verb} solved"
    end = "." if o.text.endswith(".") else ""
    return Finding(
        values=[
            ("every instance solved up to", f"{up_to} sectors"),
            (
                f"{big}-sector instances solved",
                ", ".join(inst_label(i) for i in solved_big) or "none",
            ),
        ],
        proposed=first + last + end,
    )


def _qibis_bound(a: Analysis, o: Occurrence) -> Finding:
    q = [
        r
        for r in a.d.runs
        if r.algorithm == "qibis" and r.outcome == "n/c" and r.reason == "max_iters"
    ]
    cyc = [
        r
        for r in a.d.runs
        if r.algorithm == "qibis" and r.outcome == "n/c" and r.reason == "cycle"
    ]
    vals = [
        ("Q-IBIS runs at the iteration bound", str(len(q))),
        ("Q-IBIS runs ending by recurrence", str(len(cyc))),
    ]
    if a.max_iters is None:
        return Finding(
            values=vals,
            proposed=f"In the evaluation, which had no iteration bound for the direct runs, {len(cyc)} \\gls{{qibis}} runs returned to an earlier play without an equilibrium.",
            note="The direct runs have no iteration bound in this protocol; the sentence now counts the runs that end by recurrence.",
        )
    return Finding(
        values=vals,
        proposed=f"In the evaluation, {len(q)} \\gls{{qibis}} runs reached the iteration bound without an equilibrium.",
    )


def _largest_sectors(a: Analysis, o: Occurrence) -> Finding:
    best = a.largest_solved_any()
    s = best.sectors if best else None
    return Finding(
        values=[("largest territory with a solved instance", f"{s} sectors")],
        proposed=str(s),
    )


def _largest_agents(a: Analysis, o: Occurrence) -> Finding:
    best = a.largest_solved_any()
    if best is None:
        return Finding(values=[("solved", "none")], proposed=None)
    ag = a.agents_solved_at(best.sectors)
    return Finding(
        values=[
            (
                f"agents of the solved {best.sectors}-sector instances",
                ", ".join(map(str, ag)),
            )
        ],
        proposed=and_list([words(x) for x in ag], " and "),
        note=(
            "The sentence says 'have 30 sectors and two agents'; if several agent counts are solved at that size, reword."
            if len(ag) > 1
            else None
        ),
    )


def _encoding(a: Analysis, o: Occurrence) -> Finding:
    if not a.encoding:
        return Finding(
            note="Needs --encoding (python -m experiments.encoding_times).",
            computable=False,
            values=[],
        )
    big = max(r["num_sectors"] for r in a.encoding)
    mx = max(r["build_s"] for r in a.encoding)
    mx_big = max(r["build_s"] for r in a.encoding if r["num_sectors"] == big)
    phrase = "less than a second" if mx < 1 else f"at most {mx:.1f} seconds"
    return Finding(
        values=[
            ("max time to build F, all instances", f"{mx:.2f} s"),
            (f"max on the {big}-sector territories", f"{mx_big:.2f} s"),
        ],
        proposed=phrase,
    )


def _machine(a: Analysis, o: Occurrence) -> Finding:
    vals = a.provenance_values(
        lambda s: (
            s.get("machine", {}).get("cpu_model"),
            s.get("machine", {}).get("memory_gb"),
        )
    )
    if not vals:
        return Finding(
            note="The log has no session header (examined format).", computable=False
        )
    cpu, mem = vals[0]
    note = (
        "Sessions ran on different machines: "
        + "; ".join(f"{c} {m} GB" for c, m in vals)
        if len(vals) > 1
        else None
    )
    mem_s = f"{mem:g}" if isinstance(mem, (int, float)) else str(mem)
    return Finding(
        values=[("CPU, memory", f"{cpu}, {mem_s} GB")],
        proposed=(
            f"an {cpu} with {mem_s}\\,GB of memory"
            if str(cpu)[:1].upper() in "AEIOU"
            else f"a {cpu} with {mem_s}\\,GB of memory"
        ),
        note=note,
    )


def _prov(
    label: str, key: Callable[[dict[str, Any]], Any]
) -> Callable[[Analysis, Occurrence], Finding]:
    def f(a: Analysis, o: Occurrence) -> Finding:
        vals = a.provenance_values(key)
        if not vals:
            return Finding(
                note="The log has no session header (examined format).",
                computable=False,
            )
        note = f"Several values in the log: {vals}" if len(vals) > 1 else None
        return Finding(
            values=[(label, ", ".join(map(str, vals)))],
            proposed=str(vals[0]),
            note=note,
        )

    return f


def _tool_version(a: Analysis, o: Occurrence) -> Finding:
    f = _prov("tool version", lambda s: s.get("tool", {}).get("version"))(a, o)
    commits = a.provenance_values(
        lambda s: (s.get("tool", {}).get("commit"), s.get("tool", {}).get("dirty"))
    )
    if commits:
        f.values.append(
            ("tool commit (dirty)", "; ".join(f"{c} ({d})" for c, d in commits))
        )
        if any(d for _, d in commits):
            f.note = (
                f.note or ""
            ) + " Some session ran with uncommitted changes (dirty=true)."
    return f


def _workers(a: Analysis, o: Occurrence) -> Finding:
    vals = a.provenance_values(lambda s: s.get("workers"))
    if not vals:
        return Finding(
            note="The log has no session header (examined format).", computable=False
        )
    note = (
        f"Sessions used different numbers of workers: {vals}" if len(vals) > 1 else None
    )
    return Finding(
        values=[("parallel runs (workers)", ", ".join(map(str, vals)))],
        proposed=words(int(vals[0])),
        note=note,
    )


def _variation(a: Analysis, o: Occurrence) -> Finding:
    v, n = a.variation()
    if v is None:
        return Finding(
            values=[("repeat sweep", "not given")],
            note="Needs the repeat sweep with a limit of 300 s (./reproduce.sh runs it unless --no-repeat; pass --repeat here).",
            computable=False,
        )
    limit = a.repeat.time_limit_s if a.repeat else 300
    return Finding(
        values=[
            ("pairs solved in both sweeps", str(n)),
            ("median |t1-t2|/min(t1,t2)", pct(v)),
        ],
        proposed=f"Repeating the sweep with a limit of {limit:g} seconds put the variation in wall-clock time between two runs of the same pair at a median of {pct(v)}.",
        note="Variation is defined here as |t1 - t2| / min(t1, t2); state the definition in the thesis if you keep the sentence.",
    )


def _qualitative(note: str) -> Callable[[Analysis, Occurrence], Finding]:
    def f(a: Analysis, o: Occurrence) -> Finding:
        v, n = a.variation()
        vals = [
            (
                "measured variation",
                (
                    f"{pct(v)} over {n} pairs"
                    if v is not None
                    else "no repeat sweep given"
                ),
            )
        ]
        return Finding(values=vals, note=note, computable=False)

    return f


RULES: list[Rule] = [
    # abstract and summaries
    ("abstract", _contains("solves the most instances", "CEGAR-NE solves"), _abstract),
    (
        "summary: S-IBIS most",
        _contains("solved the most instances", "without and"),
        _summary_sibis_most,
    ),
    ("summary: speed-up", _contains("its median speed-up of"), _summary_speedup),
    (
        "speed-up (chapter 7)",
        _contains(
            "pairs solved both with and without symmetry reduction, the median speed-up is"
        ),
        _speedup_main,
    ),
    (
        "speed-up (generic)",
        _contains("median speed-up from symmetry reduction"),
        _generic_speedup,
    ),
    ("CEGAR fewer", _contains("CEGAR-NE solved", "symmetry reduction"), _cegar_fewer),
    (
        "CEGAR fewer (summary)",
        _contains("CEGAR-NE solved fewer instances"),
        _cegar_fewer,
    ),
    (
        "size summary",
        _contains("Every instance with at most", "was solved by some configuration"),
        _size_summary,
    ),
    (
        "Q-IBIS bound (conclusions)",
        _contains("runs reached the iteration bound without an equilibrium"),
        _qibis_bound,
    ),
    (
        "both ends (short)",
        lambda o: o.text == "unsolved instances occur at both ends",
        _both_ends,
    ),
    (
        "qualitative (conclusions)",
        _contains("They support comparisons of solve counts"),
        _qualitative(
            "Qualitative: keep if the measured variation and the results still support it."
        ),
    ),
    # chapter 1
    (
        "largest sectors",
        _short("30", "The largest instances that we solve have"),
        _largest_sectors,
    ),
    (
        "largest agents",
        lambda o: o.prefix.rstrip().endswith("sectors and") and o.text.isalpha(),
        _largest_agents,
    ),
    ("coarse cells", _short("30", "at most"), _largest_sectors),
    # counts
    (
        "S-IBIS off",
        lambda o: o.text.isdigit()
        and o.prefix.rstrip().endswith(r"\gls{sibis} solves"),
        _count_rule("sibis", False),
    ),
    (
        "CEGAR off (ch1)",
        lambda o: o.text.isdigit()
        and o.prefix.rstrip().endswith("inner synthesiser, solves"),
        _count_rule("cegar-sibis", False),
    ),
    (
        "S-IBIS on (ch1)",
        lambda o: o.text.isdigit() and o.prefix.rstrip().endswith("the counts are"),
        _count_rule("sibis", True),
    ),
    (
        "CEGAR on (ch1)",
        lambda o: o.text.isdigit() and o.prefix.rstrip().endswith("and~"),
        _count_rule("cegar-sibis", True),
    ),
    (
        "CEGAR off (ch7)",
        lambda o: o.text.isdigit() and o.prefix.rstrip().endswith("CEGAR-NE"),
        _count_rule("cegar-sibis", False),
    ),
    (
        "IBIS off (ch7)",
        lambda o: o.text.isdigit() and o.prefix.rstrip().endswith(r"\gls{ibis}"),
        _count_rule("ibis", False),
    ),
    (
        "Q-IBIS off (ch7)",
        lambda o: o.text.isdigit() and o.prefix.rstrip().endswith(r"\gls{qibis}"),
        _count_rule("qibis", False),
    ),
    (
        "counts on (ch7)",
        lambda o: bool(re.fullmatch(r"\d+(, \d+)* and~\d+", o.text)),
        _counts_on,
    ),
    ("cactus order", _contains("keep this order at every time"), _cactus_order),
    # chapter 7
    ("encoding time", lambda o: o.text == "less than a second", _encoding),
    ("machine", _contains("with", "of memory"), _machine),
    ("tool version", _short("X.Y", "version~"), _tool_version),
    (
        "Python version",
        lambda o: o.prefix.rstrip().endswith("Python~"),
        _prov("Python version", lambda s: s.get("python", {}).get("version")),
    ),
    (
        "Z3 version",
        lambda o: o.prefix.rstrip().endswith("Z3~"),
        _prov("z3-solver version", lambda s: s.get("z3_version")),
    ),
    (
        "workers",
        lambda o: o.text.isalpha() and o.prefix.rstrip().endswith("was run once, and"),
        _workers,
    ),
    ("variation", _contains("Repeating the sweep with a limit of"), _variation),
    (
        "ties",
        _contains("We therefore base conclusions on"),
        _qualitative(
            "Qualitative: keep if the measured variation still supports treating such differences as ties."
        ),
    ),
    ("timeouts", _contains("Timeouts account for"), _timeouts_unsolved),
    ("asset", lambda o: o.text.startswith("\\input{"), _asset),
    ("appendix caption", _contains(r"\emph{t/o} marks a timeout"), _caption_markers),
    (
        "S-IBIS median",
        _contains(r"The median time of \gls{sibis} on solved instances"),
        _sibis_median,
    ),
    ("curves", _contains("differ only near their right-hand ends"), _curves_differ),
    ("well within", _contains("well within the limit"), _well_within),
    (
        "S-IBIS vs IBIS",
        _contains("with the version of Z3 that we used"),
        _sibis_vs_ibis,
    ),
    (
        "Q-IBIS fewest",
        _contains("solves the fewest instances of the three direct synthesisers"),
        _qibis_fewest,
    ),
    (
        "iterations",
        _contains("improvement iterations on every instance they solve"),
        _iterations,
    ),
    ("twice", _contains("at least twice as fast"), _twice),
    ("only", _contains("runs that are unsolved without it"), _only),
    ("per family", _contains("on the square and rectangular grids"), _per_family),
    ("discrete orbit", _contains("This is the case on"), _discrete_orbit),
    (
        "CEGAR vs S-IBIS off",
        _contains("Without symmetry reduction, CEGAR-NE solves"),
        _cegar_vs_sibis(False),
    ),
    (
        "CEGAR vs S-IBIS on",
        _contains("With symmetry reduction, CEGAR-NE solves"),
        _cegar_vs_sibis(True),
    ),
    (
        "S-IBIS only discrete",
        _contains("Of the instances that only"),
        _sibis_only_discrete,
    ),
    ("CEGAR verified", _contains("CEGAR-NE verified"), _cegar_verified),
    (
        "size",
        _contains("At", "sectors the first instances remain unsolved"),
        _size_main,
    ),
    ("both ends", _contains("unsolved instances occur at both ends."), _both_ends),
    ("certifier", _contains("It accepted all"), _certifier),
    ("at bound", _contains("every agent has reward"), _at_bound),
]


def match_rule(
    o: Occurrence,
) -> tuple[str, Callable[[Analysis, Occurrence], Finding]] | None:
    for name, pred, fn in RULES:
        try:
            if pred(o):
                return name, fn
        except Exception:  # noqa: BLE001 - a bad predicate must not stop the report
            continue
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _session_lines(a: Analysis) -> list[str]:
    out = []
    for s in a.sessions():
        tool = s.get("tool", {})
        mach = s.get("machine", {})
        out.append(
            f"- session {s.get('session')}: started {s.get('started_at')}, rcmas {tool.get('version')} "
            f"at {str(tool.get('commit'))[:12]}{' (DIRTY)' if tool.get('dirty') else ''}, "
            f"Python {s.get('python', {}).get('version')}, z3-solver {s.get('z3_version')}, "
            f"{mach.get('cpu_model')}, {mach.get('memory_gb')} GB, {s.get('workers')} workers, "
            f"protocol {s.get('protocol')}, command `{s.get('command')}`"
        )
    return out or ["- no session header (log of the examined runner)"]


def _initial_section(a: Analysis) -> list[str]:
    """Equilibria at the bound ``m``, runs that returned their initial play,
    initial-query times and stopping reasons."""
    lines = ["## Initial play, equilibria at the bound m, and stopping reasons", ""]
    k, n = a.all_m_share()
    lines.append(
        f"- equilibria (solved runs) in which every agent has reward m: {k} of {n}"
        + (f" ({pct(k / n)})" if n else "")
    )
    if not a.d.has_initial_fields:
        lines.append(
            "- the log does not record the initial play (a log of the first run "
            "or of the examined runner): no initial-query times and no count of "
            "runs that returned their initial play"
        )
    lines.append("")
    lines.append(
        "| configuration | equilibria with every reward m | returned the initial play "
        "| initial play with every reward m | median initial query (s) |"
    )
    lines.append("|---|---|---|---|---|")
    for alg, sym in a.d.configs:
        kk, nn = a.all_m_share(a.d.runs_of(alg, sym))
        ret = a.returned_initial(alg, sym)
        im = a.initial_all_m(alg, sym)
        iq = a.initial_query_times(alg, sym)
        if iq is None:
            iq_s = "--"
        else:
            med, cnt, unf = iq
            med_s = (
                "> limit"
                if med == float("inf")
                else t1(med) if med is None else f"{med:.2f}"
            )
            iq_s = f"{med_s} over {cnt} runs" + (
                f", {unf} not answered within the limit" if unf else ""
            )
        lines.append(
            f"| {config_label(alg, sym)} | {kk} of {nn} | "
            f"{'--' if ret is None else len(ret)} | "
            f"{'--' if im is None else f'{im[0]} of {im[1]}'} | {iq_s} |"
        )
    lines.append("")
    lines.append(
        "Initial-query medians per algorithm, without / with symmetry reduction (s):"
    )
    lines.append("")
    for alg in a.algs:
        cells = []
        for sym in (False, True):
            iq = a.initial_query_times(alg, sym)
            if iq is None or iq[0] is None:
                cells.append("--")
            elif iq[0] == float("inf"):
                cells.append("> limit")
            else:
                cells.append(f"{iq[0]:.2f}")
        lines.append(f"- {PLAIN.get(alg, alg)}: {cells[0]} / {cells[1]}")
    lines.append("")
    lines.append("Outcomes by the reason each run stopped:")
    lines.append("")
    table = a.outcomes_by_reason()
    keys = sorted({k for c in table.values() for k in c}, key=_outcome_order)
    lines.append("| configuration | " + " | ".join(keys) + " |")
    lines.append("|---|" + "---|" * len(keys))
    for cfg, counts in table.items():
        lines.append(
            f"| {config_label(*cfg)} | "
            + " | ".join(str(counts.get(k, 0)) for k in keys)
            + " |"
        )
    totals = {k: sum(c.get(k, 0) for c in table.values()) for k in keys}
    lines.append("| all | " + " | ".join(str(totals[k]) for k in keys) + " |")
    return lines


def _outcome_order(key: str) -> tuple[int, str]:
    order = ["ne", "ne (not certified)", "t/o", "n/c", "fail", "unk", "uns", "error"]
    head = key.split(" ", 1)[0]
    return (order.index(head) if head in order else len(order), key)


PLAIN = {
    "ibis": "IBIS",
    "sibis": "S-IBIS",
    "qibis": "Q-IBIS",
    "cegar-sibis": "CEGAR-NE",
}


def build_report(
    a: Analysis,
    occurrences: list[Occurrence],
    red_groups: list[tuple[str, int, str]],
    inputs: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    lines = ["# Numbers for the \\rerun markers of the thesis", ""]
    lines.append(
        "Generated by `python -m experiments.thesis_numbers`. Nothing in the thesis was changed."
    )
    lines.append("")
    lines.append("## Inputs")
    for k, v in inputs.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Provenance of the log")
    lines += _session_lines(a)
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    for proto in a.d.protocols or [a.protocol]:
        lines.append(f"- {protocol_text(proto)}")
    if len(a.d.protocols) > 1:
        lines.append(
            "- **the log mixes synthesis protocols; its numbers are not comparable**"
        )
    lines.append("")
    lines.append("## Data checks")
    expected = a.N * len(a.d.configs)
    lines.append(
        f"- {a.N} instances, {len(a.d.configs)} configurations, {len(a.d.runs)} runs (complete grid: {expected})"
    )
    from collections import Counter

    oc = Counter(r.outcome for r in a.d.runs)
    lines.append("- outcomes: " + ", ".join(f"{k} {v}" for k, v in oc.most_common()))
    lines.append(
        "- certifier: "
        + ", ".join(f"{k} {v}" for k, v in a.certificate_counts().items())
    )
    if a.d.recertified:
        lines.append(f"- re-certification entries matched: {a.d.recertified}")
    errors = [r for r in a.d.runs if r.outcome == "error"]
    if errors:
        lines.append(
            f"- **{len(errors)} runs ended with an error of the tool**: "
            + ", ".join(f"{r.name} ({r.reason})" for r in errors[:10])
        )
    for p in a.d.problems[:50]:
        lines.append(f"- problem: {p}")
    if len(a.d.problems) > 50:
        lines.append(f"- ... and {len(a.d.problems) - 50} more problems")
    lines.append("")
    lines.append("## Solved instances per configuration")
    lines.append("")
    lines.append("| configuration | solved | median time (s) | largest solved |")
    lines.append("|---|---|---|---|")
    for alg, sym in a.d.configs:
        big = a.d.largest_solved(alg, sym)
        lines.append(
            f"| {config_label(alg, sym)} | {a.count(alg, sym)}/{a.N} | {t1(a.median(alg, sym))} | {inst_label(big) if big else '--'} |"
        )
    lines.append("")
    lines += _initial_section(a)
    lines.append("")
    lines.append(f"## Markers ({len(occurrences)} occurrences)")
    lines.append("")
    entries: list[dict[str, Any]] = []
    for number, o in enumerate(occurrences, 1):
        rule = match_rule(o)
        pos = f", marker {o.index} of {o.count} on the line" if o.count > 1 else ""
        lines.append(f"### {number}. {o.where}{pos}")
        lines.append("")
        lines.append("Current text:")
        lines.append("")
        lines.append("```latex")
        lines.append(o.text)
        lines.append("```")
        if rule is None:
            lines.append("")
            lines.append(
                "No rule matches this marker; check it by hand against the data above."
            )
            entries.append(
                {"where": o.where, "index": o.index, "current": o.text, "rule": None}
            )
            lines.append("")
            continue
        name, fn = rule
        try:
            f = fn(a, o)
        except Exception as e:  # noqa: BLE001 - report, do not crash
            f = Finding(
                note=f"could not compute ({type(e).__name__}: {e})", computable=False
            )
        lines.append("")
        lines.append(
            f"Rule: {name}{'' if f.computable else ' (not fully computable from the logs)'}"
        )
        if f.values:
            lines.append("")
            lines.append("New values:")
            for label, value in f.values:
                lines.append(f"- {label}: {value}")
        if f.proposed is not None:
            same = " (unchanged)" if f.proposed == o.text else ""
            lines.append("")
            lines.append(f"Proposed replacement{same}:")
            lines.append("")
            lines.append("```latex")
            lines.append(f.proposed)
            lines.append("```")
        if f.note:
            lines.append("")
            lines.append(f"Note: {f.note.strip()}")
        lines.append("")
        entries.append(
            {
                "where": o.where,
                "index": o.index,
                "current": o.text,
                "rule": name,
                "computable": f.computable,
                "values": f.values,
                "proposed": f.proposed,
                "note": f.note,
            }
        )
    if red_groups:
        lines.append("## Other red markers")
        lines.append("")
        for path, line, text in red_groups:
            lines.append(
                f"- {path}:{line}: `{text}` (remove the red group once the regenerated table is in place)"
            )
        lines.append("")
    return "\n".join(lines) + "\n", entries


def facts(a: Analysis) -> dict[str, Any]:
    """The main numbers as a flat dictionary (for --json and tests)."""
    v, nv = a.variation()
    sp = a.speedups()
    solved = a.solved_runs()
    p = a.size_profile()
    out: dict[str, Any] = {
        "instances": a.N,
        "runs": len(a.d.runs),
        "solved_runs": len(solved),
        "solved_over_300s": sum(r.time_s > 300 for r in solved),
        "solved_at_bound": sum(r.all_at_bound for r in solved),
        "speedup_pairs": len(sp),
        "median_speedup": a.median_speedup(),
        "twice_as_fast": sum(s >= 2 for s, _, _ in sp),
        "twice_as_slow": sum(s <= 0.5 for s, _, _ in sp),
        "only_with_symmetry": len(a.only(True)),
        "only_without_symmetry": len(a.only(False)),
        "variation": v,
        "variation_pairs": nv,
        "all_solved_up_to": p["all_solved_up_to"],
        "first_unsolved_size": p["first_unsolved_size"],
        "discrete_orbit_instances": len(a.discrete_orbit_instances()),
        "qibis_nc_bound": sum(
            1
            for r in a.d.runs
            if r.algorithm == "qibis" and r.outcome == "n/c" and r.reason == "max_iters"
        ),
        "max_iterations_ibis_sibis": max(
            (
                r.iterations
                for r in solved
                if r.algorithm in ("ibis", "sibis") and r.iterations is not None
            ),
            default=None,
        ),
        "certificates": a.certificate_counts(),
        "protocol": a.protocol,
        "equilibria_all_m": a.all_m_share()[0],
        "equilibria": a.all_m_share()[1],
        "returned_initial": None,
        "outcomes_by_reason": {
            config_label(*cfg): counts for cfg, counts in a.outcomes_by_reason().items()
        },
    }
    returned = [a.returned_initial(alg, sym) for alg, sym in a.d.configs]
    if any(x is not None for x in returned):
        out["returned_initial"] = sum(len(x) for x in returned if x is not None)
    for alg, sym in a.d.configs:
        key = config_label(alg, sym)
        out[f"solved[{key}]"] = a.count(alg, sym)
        out[f"median[{key}]"] = a.median(alg, sym)
        ret = a.returned_initial(alg, sym)
        out[f"returned_initial[{key}]"] = None if ret is None else len(ret)
        iq = a.initial_query_times(alg, sym)
        med = None if iq is None else iq[0]
        over = med is not None and math.isinf(med)
        # JSON has no infinity: a median above the limit is null with the flag set.
        out[f"initial_query_median[{key}]"] = None if over else med
        out[f"initial_query_median_over_limit[{key}]"] = over
        out[f"initial_query_unfinished[{key}]"] = None if iq is None else iq[2]
        k, n = a.all_m_share(a.d.runs_of(alg, sym))
        out[f"equilibria_all_m[{key}]"] = k
    return out


def _cert_counts(path: str | None) -> dict[str, int] | None:
    if not path:
        return None
    counts = {"accepted": 0, "rejected": 0, "unchecked": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                st = json.loads(line).get("status", "unchecked")
                counts[st] = counts.get(st, 0) + 1
    return counts


def _read_jsonl(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.thesis_numbers", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--results", required=True, help="Result log of the 900 s sweep")
    ap.add_argument(
        "--certificates",
        default=None,
        help="Output of `rcmas certify --output` for the log",
    )
    ap.add_argument(
        "--repeat", default=None, help="Result log of the repeat sweep (300 s)"
    )
    ap.add_argument(
        "--repeat-certificates", default=None, help="Certificates of the repeat sweep"
    )
    ap.add_argument(
        "--encoding",
        default=None,
        help="Output of `python -m experiments.encoding_times`",
    )
    ap.add_argument(
        "--thesis-dir", default=None, help="Thesis checkout to scan for \\rerun markers"
    )
    ap.add_argument("--output", default=None, help="Markdown report (default: stdout)")
    ap.add_argument(
        "--json", default=None, help="Also write the findings and facts as JSON"
    )
    ap.add_argument(
        "--no-certificate-required",
        action="store_true",
        help="Count uncertified equilibria as solved (only for the examined logs)",
    )
    args = ap.parse_args(argv)
    if args.thesis_dir and not is_thesis_checkout(args.thesis_dir):
        # A wrong path would otherwise give an empty report or a new tree of
        # assets that the thesis never inputs, with exit code 0.
        ap.error(
            f"--thesis-dir {args.thesis_dir} is not a thesis checkout "
            f"(no {THESIS_MARKER} there)"
        )

    req = not args.no_certificate_required
    d = load(args.results, certificates=args.certificates, require_certificate=req)
    rep = (
        load(
            args.repeat, certificates=args.repeat_certificates, require_certificate=req
        )
        if args.repeat
        else None
    )
    enc = _read_jsonl(args.encoding)
    if enc is not None:
        enc = [r for r in enc if r.get("kind", "encoding") == "encoding"]
    a = Analysis(d, rep, _cert_counts(args.certificates), enc)
    occ = find_reruns(args.thesis_dir) if args.thesis_dir else []
    red = find_red_groups(args.thesis_dir) if args.thesis_dir else []
    inputs = {
        "results": args.results,
        "certificates": args.certificates or "(certificates in the log)",
        "repeat sweep": args.repeat or "(none)",
        "encoding times": args.encoding or "(none)",
        "thesis": args.thesis_dir or "(none: no markers scanned)",
    }
    report, entries = build_report(a, occ, red, inputs)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        unmatched = sum(1 for e in entries if e["rule"] is None)
        print(f"wrote {args.output}: {len(occ)} markers, {unmatched} without a rule")
    else:
        sys.stdout.write(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"facts": facts(a), "markers": entries}, indent=1, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
