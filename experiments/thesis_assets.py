"""Generate the LaTeX assets of the thesis evaluation chapter.

Reads the thesis-eval result files (JSONL, one record per run) and writes
the summary table, the cactus plot, the paired scatter plots, and the full
per-instance appendix table as standalone .tex files that the thesis inputs.

Usage:
    python -m experiments.thesis_assets \
        --primary experiments/thesis-eval-900.jsonl \
        --secondary experiments/thesis-eval-300.jsonl \
        --chapter-dir ../../thesis/chapters/experiment \
        --appendix-dir ../../thesis/appendices/experiment

Only run-1 records are used from either file; the secondary (300 s) file
contributes the budget-sensitivity numbers printed to stdout and nothing
else.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

ALGS = ["ibis", "sibis", "qibis", "cegar-sibis"]
ALG_LABEL = {
    "ibis": "IBIS",
    "sibis": "S-IBIS",
    "qibis": "Q-IBIS",
    "cegar-sibis": "CEGAR",
}
ALG_COLOR = {
    "ibis": "blue!70!black",
    "sibis": "green!55!black",
    "qibis": "orange!90!black",
    "cegar-sibis": "violet",
}
FAMILIES = ["symmetric", "local_symmetry", "disconnected", "asymmetric"]
FAM_STYLE = {
    "symmetric": ("blue!70!black", "o"),
    "local_symmetry": ("green!55!black", "triangle*"),
    "disconnected": ("red!75!black", "square*"),
    "asymmetric": ("violet", "diamond*"),
}
FAM_LABEL = {
    "symmetric": "symmetric",
    "local_symmetry": "L-shaped",
    "disconnected": "disconnected",
    "asymmetric": "random",
}


def load(path: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if r["run"] == 1]


def gridname(r: dict) -> str:
    return os.path.basename(r["grid_path"]).replace(".txt", "")


def family(r: dict) -> str:
    return r["grid_path"].split("/grids/")[1].split("/")[0]


def cell(r: dict) -> str:
    """Appendix-table cell for one run."""
    if r["status"] == "timeout":
        return "t/o"
    if r["reason"] == "unsat":
        return "uns"
    if r["reason"] == "max_iters":
        return "n/c"
    return f"{r['time_s']:.1f}"


def write_summary(d: list[dict], outdir: str) -> None:
    lines = [
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Algorithm & Symmetry & NE found & Timeouts & Median time (s) & Largest instance solved \\",
        r"\midrule",
    ]
    for alg in ALGS:
        for sym in [False, True]:
            rs = [r for r in d if r["algorithm"] == alg and r["symmetry"] == sym]
            ok = [r for r in rs if r["found_ne"]]
            to = [r for r in rs if r["status"] == "timeout"]
            med = statistics.median(r["time_s"] for r in ok)
            big = max(ok, key=lambda r: (r["num_sectors"], r["num_agents"]))
            label = ALG_LABEL[alg] if not sym else ""
            lines.append(
                f"{label} & {'on' if sym else 'off'} & {len(ok)}/70 & {len(to)}"
                f" & {med:.1f} & {gridname(big)} ($|T|={big['num_sectors']}$, $n={big['num_agents']}$) \\\\"
            )
        if alg != ALGS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(outdir, "tab-summary.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_cactus(d: list[dict], outdir: str) -> None:
    plots = []
    legend = []
    for alg in ALGS:
        for sym in [False, True]:
            times = sorted(
                r["time_s"]
                for r in d
                if r["algorithm"] == alg and r["symmetry"] == sym and r["found_ne"]
            )
            coords = " ".join(f"({i},{t:.3g})" for i, t in enumerate(times, 1))
            style = f"color={ALG_COLOR[alg]}, mark=none, thick"
            if sym:
                style += ", dashed"
            plots.append(f"\\addplot[{style}] coordinates {{ {coords} }};")
            legend.append(ALG_LABEL[alg] + ("+sym" if sym else ""))
    body = "\n".join(plots)
    legend_str = ", ".join(legend)
    tex = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.92\textwidth, height=0.58\textwidth,
    ymode=log, xmin=0, xmax=62, ymin=0.04, ymax=3000,
    xlabel={{instances solved}}, ylabel={{wall-clock time (s)}},
    grid=major, grid style={{gray!20}},
    legend pos=north west, legend columns=2,
    legend style={{font=\footnotesize, fill=white, fill opacity=0.85, draw opacity=1, text opacity=1}},
    tick label style={{font=\footnotesize}}, label style={{font=\small}},
]
{body}
\addplot[gray, dotted, thick] coordinates {{ (0,300) (62,300) }};
\node[gray, font=\footnotesize, anchor=north east] at (axis cs:61.5,280) {{300\,s budget}};
\legend{{{legend_str}}}
\end{{axis}}
\end{{tikzpicture}}
"""
    with open(os.path.join(outdir, "fig-cactus.tex"), "w") as f:
        f.write(tex)


def _scatter_axis(points_by_fam: dict, xlabel: str, ylabel: str) -> str:
    plots = []
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
    legend = ", ".join(FAM_LABEL[f] for f in FAMILIES if points_by_fam.get(f))
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth, height=\linewidth,
    xmode=log, ymode=log,
    xmin=0.04, xmax=1500, ymin=0.04, ymax=1500,
    xlabel={{{xlabel}}}, ylabel={{{ylabel}}},
    grid=major, grid style={{gray!20}},
    legend pos=north west,
    legend style={{font=\scriptsize, fill=white, fill opacity=0.85, draw opacity=1, text opacity=1}},
    tick label style={{font=\footnotesize}}, label style={{font=\footnotesize}},
]
\addplot[black!60, forget plot] coordinates {{ (0.04,0.04) (1500,1500) }};
{body}
\legend{{{legend}}}
\end{{axis}}
\end{{tikzpicture}}
"""


def write_scatters(d: list[dict], outdir: str) -> None:
    ix = {(gridname(r), r["num_agents"], r["algorithm"], r["symmetry"]): r for r in d}
    # (a) symmetry off vs on, pooled over algorithms
    sym_pts: dict[str, list] = defaultdict(list)
    for (g, n, a, s), r in ix.items():
        if s:
            continue
        r2 = ix.get((g, n, a, True))
        if r2 and r["found_ne"] and r2["found_ne"]:
            sym_pts[family(r)].append((r["time_s"], r2["time_s"]))
    # (b) S-IBIS vs CEGAR, paired within the same symmetry setting
    cg_pts: dict[str, list] = defaultdict(list)
    for (g, n, a, s), r in ix.items():
        if a != "sibis":
            continue
        r2 = ix.get((g, n, "cegar-sibis", s))
        if r2 and r["found_ne"] and r2["found_ne"]:
            cg_pts[family(r)].append((r["time_s"], r2["time_s"]))
    with open(os.path.join(outdir, "fig-scatter-sym.tex"), "w") as f:
        f.write(_scatter_axis(sym_pts, "time without symmetry (s)", "time with symmetry (s)"))
    with open(os.path.join(outdir, "fig-scatter-cegar.tex"), "w") as f:
        f.write(_scatter_axis(cg_pts, "S-IBIS time (s)", "CEGAR time (s)"))


def write_full_table(d: list[dict], outdir: str) -> None:
    ix = {(gridname(r), r["num_agents"], r["algorithm"], r["symmetry"]): r for r in d}
    insts = sorted(
        {(gridname(r), r["num_sectors"], r["num_agents"], r["horizon"]) for r in d},
        key=lambda x: (x[1], x[0], x[2]),
    )
    lines = [
        r"\begin{longtable}{@{}lrrr rrrrrrrr@{}}",
        r"\caption[Per-instance synthesis times]{Wall-clock synthesis time in seconds"
        r" for every instance and configuration at the 900-second budget."
        r" \emph{t/o} marks a timeout, \emph{n/c} a run that stopped at the iteration"
        r" cap without converging, and \emph{uns} a run whose candidate-restricted"
        r" encoding was unsatisfiable. Paired columns are without and with symmetry"
        r" reduction.}\label{tab:exp:full}\\",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{IBIS} & \multicolumn{2}{c}{S-IBIS}"
        r" & \multicolumn{2}{c}{Q-IBIS} & \multicolumn{2}{c}{CEGAR} \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(l){11-12}",
        r"Grid & $|T|$ & $n$ & $m$ & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym}"
        r" & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym}"
        r" & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym}"
        r" & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{IBIS} & \multicolumn{2}{c}{S-IBIS}"
        r" & \multicolumn{2}{c}{Q-IBIS} & \multicolumn{2}{c}{CEGAR} \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(l){11-12}",
        r"Grid & $|T|$ & $n$ & $m$ & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym}"
        r" & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym}"
        r" & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym}"
        r" & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{sym} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for g, sec, n, h in insts:
        cells = []
        for alg in ALGS:
            for sym in [False, True]:
                cells.append(cell(ix[(g, n, alg, sym)]))
        lines.append(f"{g} & {sec} & {n} & {h} & " + " & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    with open(os.path.join(outdir, "tab-full.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def budget_report(primary: list[dict], secondary: list[dict]) -> None:
    k = lambda r: r["name"]
    p = {k(r): r for r in primary}
    s = {k(r): r for r in secondary}
    gained = [n for n in p if p[n]["found_ne"] and n in s and not s[n]["found_ne"]]
    over = [r for r in primary if r["found_ne"] and r["time_s"] > 300]
    print(f"runs solved at 900 s but not 300 s: {len(gained)} of {len(p)}")
    print(f"solved runs needing more than 300 s: {len(over)} of {sum(1 for r in primary if r['found_ne'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="900 s JSONL (used for all assets)")
    ap.add_argument("--secondary", help="300 s JSONL (budget-sensitivity report only)")
    ap.add_argument("--chapter-dir", required=True)
    ap.add_argument("--appendix-dir", required=True)
    args = ap.parse_args()

    d = load(args.primary)
    write_summary(d, args.chapter_dir)
    write_cactus(d, args.chapter_dir)
    write_scatters(d, args.chapter_dir)
    write_full_table(d, args.appendix_dir)
    if args.secondary:
        budget_report(d, load(args.secondary))
    print("assets written")


if __name__ == "__main__":
    main()
