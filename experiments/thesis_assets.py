"""Generate the LaTeX assets of thesis Chapter 7 and Appendix A.

Reads a result log of :mod:`experiments.runner` (and, optionally, the
re-certification file of ``rcmas certify --output``) and writes the files
that the thesis inputs, at the paths the thesis inputs them from:

====================================================  ===================
file                                                   thesis element
====================================================  ===================
``chapters/experiment/tab-summary.tex``                Table 7.1
``chapters/experiment/fig-cactus.tex``                 Figure 7.2
``chapters/experiment/fig-scatter-sym.tex``            Figure 7.3(a)
``chapters/experiment/fig-scatter-cegar.tex``          Figure 7.3(b)
``appendices/experiment/tab-full.tex``                 Table A.1
====================================================  ===================

A run counts as solved only if the certifier accepted its equilibrium
(:mod:`experiments.results`).  Usage::

    python -m experiments.thesis_assets \\
        --results results/thesis-eval-900.jsonl \\
        --certificates results/thesis-eval-900.cert.jsonl \\
        --out-dir results/thesis-assets \\
        [--thesis-dir ../../thesis]

``--out-dir`` receives the files under the relative paths above;
``--thesis-dir`` also writes them into a thesis checkout and checks that the
thesis inputs exactly these files.  ``--chapter-dir`` and
``--appendix-dir`` (the interface of the examined version) write the
chapter and appendix files into two given directories.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from .results import (
    ALG_LABEL,
    ALGS,
    FAMILIES,
    FAMILY_LABEL,
    THESIS_MARKER,
    Results,
    Run,
    is_thesis_checkout,
    load,
)

CHAPTER = "chapters/experiment"
APPENDIX = "appendices/experiment"
ASSETS = {
    "tab-summary": CHAPTER,
    "fig-cactus": CHAPTER,
    "fig-scatter-sym": CHAPTER,
    "fig-scatter-cegar": CHAPTER,
    "tab-full": APPENDIX,
}
"""Asset name -> directory of the thesis that inputs it."""

ALG_COLOR = {
    "ibis": "blue!70!black",
    "sibis": "green!55!black",
    "qibis": "orange!90!black",
    "cegar-sibis": "violet",
}
FAM_STYLE = {
    "symmetric": ("blue!70!black", "o"),
    "local_symmetry": ("green!55!black", "triangle*"),
    "disconnected": ("red!75!black", "square*"),
    "asymmetric": ("violet", "diamond*"),
}

CELL_MARK = {
    "t/o": "t/o",
    "n/c": "n/c",
    "unk": "unk",
    "fail": "fail",
    "uns": "uns",
    "error": "err",
}
"""Appendix-table marker of every unsolved outcome."""

MARK_TEXT = {
    "t/o": "a timeout",
    "n/c": "a run that reached the iteration bound or returned to an earlier play",
    "unk": "a run in which Z3 answered a query with unknown",
    "fail": r"a CEGAR-NE run that ended with \textsc{Failure}",
    "uns": "a run whose initial query had no satisfying play",
    "err": "a run that stopped with an error of the tool",
    "rej": "an equilibrium that the certifier rejected",
    "unc": "an equilibrium that the certifier could not check in time",
    "--": "a run missing from the log",
}
"""What each marker of the appendix table stands for (caption text)."""
MARK_ORDER = ("t/o", "n/c", "unk", "fail", "uns", "err", "rej", "unc", "--")


def _algs(d: Results) -> list[str]:
    present = d.algorithms
    return [a for a in ALGS if a in present] + [a for a in present if a not in ALGS]


def cell(r: Run | None) -> str:
    """Appendix-table cell for one run."""
    if r is None:
        return "--"
    if r.solved:
        return f"{r.time_s:.1f}"
    if r.found_ne:
        return "rej" if r.certificate == "rejected" else "unc"
    return CELL_MARK.get(r.outcome, r.outcome)


def _fmt_count(x: float) -> str:
    """A number for a pgfplots option, without exponent notation for 0.001 <= x < 1e6."""
    return f"{x:.6g}"


# ---------------------------------------------------------------------------
# Table 7.1
# ---------------------------------------------------------------------------


def summary_table(d: Results) -> str:
    total = len(d.instances)
    lines = [
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Algorithm & Symmetry & NE found & Unsolved & Median time (s) & Largest instance solved \\",
        r"\midrule",
    ]
    algs = _algs(d)
    for alg in algs:
        for sym in (False, True):
            if (alg, sym) not in d.configs:
                continue
            ok = d.solved_of(alg, sym)
            med = d.median_time(alg, sym)
            big = d.largest_solved(alg, sym)
            label = ALG_LABEL.get(alg, alg) if not sym else ""
            med_s = f"{med:.1f}" if med is not None else "--"
            big_s = (
                f"{big.grid} ($|T|={big.sectors}$, $n={big.agents}$)"
                if big is not None
                else "--"
            )
            lines.append(
                f"{label} & {'on' if sym else 'off'} & {len(ok)}/{total} & {total - len(ok)}"
                f" & {med_s} & {big_s} \\\\"
            )
        if alg != algs[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figure 7.2
# ---------------------------------------------------------------------------


def cactus_plot(d: Results, mark_time: float | None = 300.0) -> str:
    plots = []
    legend = []
    most = 0
    shortest = math.inf
    for alg in _algs(d):
        for sym in (False, True):
            if (alg, sym) not in d.configs:
                continue
            times = sorted(r.time_s for r in d.solved_of(alg, sym))
            most = max(most, len(times))
            if times:
                shortest = min(shortest, times[0])
            coords = " ".join(f"({i},{t:.3g})" for i, t in enumerate(times, 1))
            style = f"color={ALG_COLOR.get(alg, 'black')}, mark=none, thick"
            if sym:
                style += ", dashed"
            plots.append(f"\\addplot[{style}] coordinates {{ {coords} }};")
            legend.append(ALG_LABEL.get(alg, alg) + ("+sym" if sym else ""))
    xmax = max(10, most + 4)
    ymin = (
        0.04
        if shortest >= 0.04 or shortest is math.inf
        else 10 ** math.floor(math.log10(max(shortest, 1e-3)))
    )
    ymax = max(3000.0, d.time_limit_s * 3)
    body = "\n".join(plots)
    legend_str = ", ".join(legend)
    mark = ""
    if mark_time:
        label = f"{mark_time:g}\\,s"
        mark = (
            f"\\addplot[gray, dotted, thick, forget plot] coordinates {{ (0,{mark_time:g}) ({xmax},{mark_time:g}) }};\n"
            f"\\node[gray, font=\\footnotesize, anchor=north east] at (axis cs:{xmax - 0.5:g},{mark_time * 0.93:.3g}) {{{label}}};\n"
        )
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.92\textwidth, height=0.58\textwidth,
    ymode=log, xmin=0, xmax={xmax}, ymin={_fmt_count(ymin)}, ymax={_fmt_count(ymax)},
    xlabel={{instances solved}}, ylabel={{wall-clock time (s)}},
    grid=major, grid style={{gray!20}},
    legend pos=north west, legend columns=2,
    legend style={{font=\footnotesize, fill=white, fill opacity=0.85, draw opacity=1, text opacity=1}},
    tick label style={{font=\footnotesize}}, label style={{font=\small}},
]
{body}
{mark}\legend{{{legend_str}}}
\end{{axis}}
\end{{tikzpicture}}
"""


# ---------------------------------------------------------------------------
# Figure 7.3
# ---------------------------------------------------------------------------


def _scatter_axis(
    points_by_fam: dict[str, list[tuple[float, float]]], xlabel: str, ylabel: str
) -> str:
    plots = []
    lo = 0.04
    for pts in points_by_fam.values():
        for x, y in pts:
            lo = min(lo, x, y)
    if lo < 0.04:
        lo = 10 ** math.floor(math.log10(max(lo, 1e-3)))
    hi = 1500
    for fam in FAMILIES:
        pts = points_by_fam.get(fam, [])
        if not pts:
            continue
        color, mark = FAM_STYLE[fam]
        coords = " ".join(f"({x:.3g},{y:.3g})" for x, y in pts)
        plots.append(
            f"\\addplot[only marks, mark={mark}, mark size=1.6pt, color={color}, "
            f"fill opacity=0.55] coordinates {{ {coords} }};"
        )
    body = "\n".join(plots)
    legend = ", ".join(FAMILY_LABEL[f] for f in FAMILIES if points_by_fam.get(f))
    legend_line = f"\\legend{{{legend}}}\n" if legend else ""
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth, height=\linewidth,
    xmode=log, ymode=log,
    xmin={_fmt_count(lo)}, xmax={hi}, ymin={_fmt_count(lo)}, ymax={hi},
    xlabel={{{xlabel}}}, ylabel={{{ylabel}}},
    grid=major, grid style={{gray!20}},
    legend pos=north west,
    legend style={{font=\scriptsize, fill=white, fill opacity=0.85, draw opacity=1, text opacity=1}},
    tick label style={{font=\footnotesize}}, label style={{font=\footnotesize}},
]
\addplot[black!60, forget plot] coordinates {{ ({_fmt_count(lo)},{_fmt_count(lo)}) ({hi},{hi}) }};
{body}
{legend_line}\end{{axis}}
\end{{tikzpicture}}
"""


def symmetry_pairs(d: Results) -> dict[str, list[tuple[float, float]]]:
    """(time without, time with) symmetry reduction, per family, for pairs both solve."""
    pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for inst in d.instances:
        for alg in _algs(d):
            r0, r1 = d.get(inst, alg, False), d.get(inst, alg, True)
            if r0 and r1 and r0.solved and r1.solved:
                pts[r0.family].append((r0.time_s, r1.time_s))
    return pts


def cegar_pairs(d: Results) -> dict[str, list[tuple[float, float]]]:
    """(S-IBIS time, CEGAR time), per family, within one symmetry setting."""
    pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for inst in d.instances:
        for sym in (False, True):
            r0, r1 = d.get(inst, "sibis", sym), d.get(inst, "cegar-sibis", sym)
            if r0 and r1 and r0.solved and r1.solved:
                pts[r0.family].append((r0.time_s, r1.time_s))
    return pts


def scatter_sym(d: Results) -> str:
    return _scatter_axis(
        symmetry_pairs(d), "time without symmetry (s)", "time with symmetry (s)"
    )


def scatter_cegar(d: Results) -> str:
    return _scatter_axis(cegar_pairs(d), "S-IBIS time (s)", "CEGAR time (s)")


# ---------------------------------------------------------------------------
# Table A.1
# ---------------------------------------------------------------------------


def _marks_sentence(marks: set[str]) -> str:
    """``\\emph{t/o} marks a timeout, \\emph{n/c} a run that ..., and ...``."""
    present = [m for m in MARK_ORDER if m in marks]
    parts = []
    for i, m in enumerate(present):
        verb = " marks " if i == 0 else " "
        parts.append(rf"\emph{{{m}}}{verb}{MARK_TEXT[m]}")
    if not parts:
        return ""
    if len(parts) == 1:
        text = parts[0]
    elif len(parts) == 2:
        text = f"{parts[0]} and {parts[1]}"
    else:
        text = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return " " + text + "."


def _caption(d: Results, marks: set[str]) -> str:
    return (
        r"\caption[Per-instance synthesis times]{Wall-clock synthesis time in seconds"
        rf" for every instance and configuration, with a limit of {d.time_limit_s:g} seconds per run,"
        r" from the result logs of the archived version of the tool~\cite{rcmas2026zenodo}."
        + _marks_sentence(marks)
        + r" Paired columns are without and with symmetry reduction.}\label{tab:exp:full}\\"
    )


def full_table(d: Results) -> str:
    algs = _algs(d)
    rows = []
    marks: set[str] = set()
    for inst in d.instances:
        info = d.instance_info(inst)
        cells = []
        for alg in algs:
            for sym in (False, True):
                c = cell(d.get(inst, alg, sym))
                if not re.fullmatch(r"[0-9.]+", c):
                    marks.add(c)
                cells.append(c)
        rows.append(
            f"{info.grid} & {info.sectors} & {info.agents} & {info.horizon} & "
            + " & ".join(cells)
            + r" \\"
        )
    ncols = 2 * len(algs)
    spec = "@{}lrrr " + "r" * ncols + "@{}"
    groups = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{ALG_LABEL.get(a, a)}}}" for a in algs
    )
    rules = "".join(
        rf"\cmidrule({'l' if i == len(algs) - 1 else 'lr'}){{{5 + 2 * i}-{6 + 2 * i}}}"
        for i in range(len(algs))
    )
    sub = " & ".join(r"\multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym}" for _ in algs)
    head = [
        r"\toprule",
        rf" & & & & {groups} \\",
        rules,
        rf"Grid & $|T|$ & $n$ & $m$ & {sub} \\",
        r"\midrule",
    ]
    lines = [rf"\begin{{longtable}}{{{spec}}}", _caption(d, marks)]
    lines += (
        head
        + [r"\endfirsthead"]
        + head
        + [r"\endhead", r"\bottomrule", r"\endlastfoot"]
    )
    lines += rows
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def render(d: Results, mark_time: float | None = 300.0) -> dict[str, str]:
    """Asset name -> LaTeX source."""
    return {
        "tab-summary": summary_table(d),
        "fig-cactus": cactus_plot(d, mark_time),
        "fig-scatter-sym": scatter_sym(d),
        "fig-scatter-cegar": scatter_cegar(d),
        "tab-full": full_table(d),
    }


def write_tree(assets: dict[str, str], root: str | Path) -> list[Path]:
    """Write every asset to ``root/<thesis dir>/<name>.tex``; returns the paths."""
    written = []
    for name, text in assets.items():
        path = Path(root) / ASSETS[name] / f"{name}.tex"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


_INPUT = re.compile(r"\\input\{([^}]+)\}")


def thesis_inputs(thesis_dir: str | Path) -> set[str]:
    """Files that the experiment chapter and appendix of a thesis checkout input."""
    found: set[str] = set()
    for sub in (CHAPTER, APPENDIX):
        main = Path(thesis_dir) / sub / "main.tex"
        if not main.is_file():
            continue
        for line in main.read_text(encoding="utf-8").splitlines():
            code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
            for m in _INPUT.finditer(code):
                target = m.group(1)
                found.add(target[:-4] if target.endswith(".tex") else target)
    return found


def check_thesis_inputs(thesis_dir: str | Path) -> tuple[set[str], set[str]]:
    """``(missing, extra)``: inputs the generator does not write, and assets the thesis does not input."""
    expected = {f"{ASSETS[n]}/{n}" for n in ASSETS}
    used = thesis_inputs(thesis_dir)
    return used - expected, expected - used


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.thesis_assets", description=__doc__.split("\n")[0]
    )
    ap.add_argument(
        "--results",
        "--primary",
        dest="results",
        required=True,
        help="Result log (JSON lines)",
    )
    ap.add_argument(
        "--certificates",
        default=None,
        help="Output of `rcmas certify --output` for the log",
    )
    ap.add_argument("--secondary", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--out-dir", default=None, help="Write the assets below this directory"
    )
    ap.add_argument(
        "--thesis-dir", default=None, help="Also write them into this thesis checkout"
    )
    ap.add_argument(
        "--chapter-dir",
        default=None,
        help="Write the chapter assets into this directory",
    )
    ap.add_argument(
        "--appendix-dir",
        default=None,
        help="Write the appendix table into this directory",
    )
    ap.add_argument(
        "--mark-time",
        type=float,
        default=300.0,
        help="Time marked by the dotted line of the cactus plot (default 300; 0 for none)",
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

    if not (args.out_dir or args.thesis_dir or args.chapter_dir or args.appendix_dir):
        ap.error("give --out-dir, --thesis-dir or --chapter-dir/--appendix-dir")
    d = load(
        args.results,
        certificates=args.certificates,
        require_certificate=not args.no_certificate_required,
    )
    for p in d.problems:
        print(f"warning: {p}", file=sys.stderr)
    assets = render(d, args.mark_time or None)
    if args.out_dir:
        for path in write_tree(assets, args.out_dir):
            print(f"wrote {path}")
    if args.thesis_dir:
        missing, extra = check_thesis_inputs(args.thesis_dir)
        for path in write_tree(assets, args.thesis_dir):
            print(f"wrote {path}")
        for m in sorted(missing):
            print(
                f"warning: the thesis inputs {m}, which this script does not generate",
                file=sys.stderr,
            )
        for e in sorted(extra):
            print(f"warning: the thesis does not input {e}", file=sys.stderr)
    for target, names in ((args.chapter_dir, CHAPTER), (args.appendix_dir, APPENDIX)):
        if target:
            os.makedirs(target, exist_ok=True)
            for name, text in assets.items():
                if ASSETS[name] == names:
                    path = Path(target) / f"{name}.tex"
                    path.write_text(text, encoding="utf-8")
                    print(f"wrote {path}")
    n_solved = sum(r.solved for r in d.runs)
    print(
        f"{len(d.instances)} instances, {len(d.runs)} runs, {n_solved} solved (certified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
