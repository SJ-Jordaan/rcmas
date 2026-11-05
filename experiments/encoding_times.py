"""Time the construction of the formula F (thesis Section 7.1).

Section 7.1 states how long the tool takes to build the cached formula
``F`` of Section 4.2 on the benchmark territories.  This script builds ``F``
with :func:`rcmas.smt_solve.build_base_encoding` for every instance of a
configuration, ``--repeats`` times each in one process, and writes one JSON
line per instance with the smallest and the largest time, after a session
header with the provenance of :mod:`experiments.provenance`::

    python -m experiments.encoding_times --config experiments/thesis-eval.toml \\
        --output results/encoding-build.jsonl

:mod:`experiments.thesis_numbers` reads the output (``--encoding``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config, load_protocol
from .provenance import collect, relative_to_tool


def measure(config: str, repeats: int = 3) -> list[dict]:
    from rcmas.model import Territory
    from rcmas.smt_solve import build_base_encoding

    protocol = load_protocol(config)
    seen: set[tuple[str, int]] = set()
    rows = []
    for sc in load_config(config):
        key = (sc.grid_path, sc.num_agents)
        if key in seen:
            continue
        seen.add(key)
        with open(sc.grid_path, encoding="utf-8") as f:
            territory = Territory.from_ascii(f)
        n = sc.num_agents
        demands = None if protocol.demand == 0 else tuple([protocol.demand] * n)
        times = []
        size = 0
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            base = build_base_encoding(
                territory=territory, num_agents=n, horizon=sc.horizon, demands=demands
            )
            times.append(time.perf_counter() - t0)
            size = len(base.constraints)
        rows.append(
            {
                "kind": "encoding",
                "grid_path": relative_to_tool(sc.grid_path),
                "num_sectors": len(territory),
                "num_agents": n,
                "horizon": sc.horizon,
                "constraints": size,
                "build_s": round(max(times), 4),
                "build_min_s": round(min(times), 4),
                "repeats": len(times),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.encoding_times", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--config", default="experiments/thesis-eval.toml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)
    rows = measure(args.config, args.repeats)
    header = collect(
        config_path=args.config, extra={"kind": "session", "purpose": "encoding_times"}
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    worst = max(rows, key=lambda r: r["build_s"])
    print(
        f"built F for {len(rows)} instances; slowest {worst['grid_path']} n={worst['num_agents']}: "
        f"{worst['build_s']:.3f} s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
