"""Outputs of an earlier sweep in the results directory (used by ``reproduce.sh``).

The sweeps of ``reproduce.sh`` resume their logs.  A log written under
another synthesis protocol (iteration bounds or initial queries) cannot be
resumed: the runner refuses it (:class:`experiments.runner.ProtocolMismatchError`).
This happens to an author who ran the first protocol of the thesis
evaluation (``N = 30``, initial play from ``smt(F)``) and now runs the
revised one.  Before the sweep, ``reproduce.sh`` calls this module, which

* does nothing if no log of the results directory has another protocol;
* otherwise prints which logs differ and how to keep them, and exits with
  2, unless ``--archive`` is given;
* with ``--archive`` moves each log of another protocol and its
  certificates into a subdirectory (``first-run-n30`` for a log of the
  first protocol), writes a README there, and exits with 0; the first run
  is kept for transparency.  The encoding times, the assets and the numbers
  report (``--outputs``) are derived from the main log and move with it.
  A log that can be resumed, or that differs only in its setup, stays in
  place with its certificates, so that its sweep resumes;
* with ``--discard`` (the smoke test, whose outputs mean nothing) deletes
  the outputs of an earlier smoke run that the sweep could not resume.

Usage::

    python -m experiments.previous --config experiments/thesis-eval.toml \\
        --results-dir results --log results/thesis-eval-900.jsonl \\
        --repeat-log results/thesis-eval-300.jsonl --repeat-limit 300 \\
        --outputs results/encoding-build.jsonl ... [--archive | --discard]

``--certificates`` and ``--repeat-certificates`` default to the log name
with ``.cert.jsonl`` for ``.jsonl``, the names that ``reproduce.sh`` uses.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import load_protocol, normalise_protocol
from .runner import ResumeCheck, check_resume, describe_protocol, read_log

FIRST_RUN_DIR = "first-run-n30"
"""Where ``--archive`` puts the outputs of a sweep of the first protocol."""


def is_first_protocol(protocol: dict[str, Any] | None) -> bool:
    """The protocol of the first run: ``N = 30`` everywhere, initial play from ``smt(F)``."""
    p = normalise_protocol(protocol) or {}
    return (
        p.get("max_iters") == 30
        and p.get("initial") == "satisfying"
        and p.get("cegar_inner_max_iters") == 30
        and p.get("cegar_discrete_max_iters") == 30
        and p.get("cegar_inner_initial") == "satisfying"
    )


def certificates_of(log: Path) -> Path:
    """The certificates that ``reproduce.sh`` writes for *log*: ``X.jsonl`` -> ``X.cert.jsonl``."""
    name = log.name[: -len(".jsonl")] if log.name.endswith(".jsonl") else log.name
    return log.with_name(name + ".cert.jsonl")


def log_protocols(log: Path) -> list[dict[str, Any]]:
    """The protocols recorded in the session headers of *log*."""
    sessions, _, _ = read_log(log)
    out: list[dict[str, Any]] = []
    for s in sessions:
        p = normalise_protocol(s.get("protocol")) or {}
        if p not in out:
            out.append(p)
    return out


def archive_name(logs: list[Path]) -> str:
    """The subdirectory for the outputs of *logs*: ``first-run-n30`` for the first protocol."""
    protocols = [p for log in logs for p in log_protocols(log)]
    if protocols and all(is_first_protocol(p) for p in protocols):
        return FIRST_RUN_DIR
    sessions = [s for log in logs for s in read_log(log)[0]]
    started = str(sessions[0].get("started_at", "")) if sessions else ""
    stamp = started[:10] or time.strftime("%Y-%m-%d")
    return f"previous-run-{stamp}"


def _readme(dest: Path, moved: list[Path], logs: list[Path], config: str) -> str:
    lines = [f"# Outputs of an earlier sweep ({dest.name})", ""]
    first = dest.name == FIRST_RUN_DIR
    if first:
        lines += [
            "These are the outputs of the first run of the evaluation of Chapter 7 of",
            "the thesis. Its protocol bounded every synthesiser by N = 30 iterations,",
            "the inner S-IBIS of CEGAR-NE included, and took the initial play of every",
            "synthesis call from smt(F).",
            "",
            f"The revised protocol (`{config}`) replaced it: no iteration bound for",
            "the direct runs of IBIS, S-IBIS and Q-IBIS, the initial play from the",
            "collectively optimal query smt(F, f), and for CEGAR-NE the bound 30 and",
            "the initial play from smt(F) of the abstract game at a coarse partition,",
            "and no bound and smt(F, f) at the discrete partition. The first run",
            "is kept for transparency; `experiments/thesis-eval-n30.toml` repeats its",
            "protocol, and the scripts of `experiments/` read these logs, for example",
            "`python -m experiments.thesis_numbers --results",
            f"{dest.parent.name}/{dest.name}/thesis-eval-900.jsonl --certificates",
            f"{dest.parent.name}/{dest.name}/thesis-eval-900.cert.jsonl`.",
        ]
    else:
        lines += [
            "These outputs were written under another protocol than",
            f"`{config}` and were moved here so that the sweep could start a new log.",
        ]
    lines += [
        "",
        f"Moved here by `./reproduce.sh --archive-previous` on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}.",
        "",
        "## Sessions",
        "",
    ]
    for log in logs:
        sessions, runs, _ = read_log(dest / log.name)
        lines.append(f"- `{log.name}`: {len(runs)} runs")
        for s in sessions:
            tool = s.get("tool", {})
            mach = s.get("machine", {})
            lines.append(
                f"  - session {s.get('session')}, started {s.get('started_at')}, "
                f"rcmas {tool.get('version')} at {str(tool.get('commit'))[:12]}"
                f"{' (dirty)' if tool.get('dirty') else ''}, z3-solver "
                f"{s.get('z3_version')}, {mach.get('cpu_model')}, "
                f"{s.get('workers')} workers; {describe_protocol(s.get('protocol') or {})}"
            )
    lines += ["", "## Files", ""]
    lines += [f"- `{p.name}`" for p in moved]
    return "\n".join(lines) + "\n"


def archive(
    results_dir: Path,
    logs: list[Path],
    outputs: list[Path],
    *,
    config: str,
    dest_name: str | None = None,
) -> Path:
    """Move *logs* and the existing *outputs* into a subdirectory of *results_dir*.

    Raises :class:`FileExistsError` if the subdirectory exists, so that an
    earlier archive is never overwritten.
    """
    dest = results_dir / (dest_name or archive_name(logs))
    if dest.exists():
        raise FileExistsError(
            f"{dest} exists; move it away or remove it before archiving again"
        )
    dest.mkdir(parents=True)
    moved = []
    for p in [*logs, *outputs]:
        if p.exists() and p.resolve() != dest.resolve():
            target = dest / p.name
            shutil.move(str(p), str(target))
            moved.append(target)
    (dest / "README.md").write_text(
        _readme(dest, moved, logs, config), encoding="utf-8"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.previous", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--log", required=True, help="Log of the main sweep")
    ap.add_argument(
        "--certificates",
        default=None,
        help="Certificates of the main log (default: the log name with .cert.jsonl)",
    )
    ap.add_argument("--repeat-log", default=None, help="Log of the repeat sweep")
    ap.add_argument(
        "--repeat-certificates",
        default=None,
        help="Certificates of the repeat log (default: its name with .cert.jsonl)",
    )
    ap.add_argument("--repeat-limit", type=int, default=None)
    ap.add_argument(
        "--outputs",
        nargs="*",
        default=[],
        help="Files and directories derived from the main log (encoding times, "
        "assets, numbers report); they move only with it",
    )
    ap.add_argument(
        "--hint",
        default="",
        help="Options to repeat in the suggested ./reproduce.sh command",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--archive",
        action="store_true",
        help="Move the outputs of another protocol into a subdirectory",
    )
    mode.add_argument(
        "--discard",
        action="store_true",
        help="Delete outputs that the sweep cannot resume (smoke test only)",
    )
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    protocol = load_protocol(args.config)
    main_log = Path(args.log)
    sweeps: list[tuple[Path, Any, Path]] = [
        (main_log, protocol, Path(args.certificates or certificates_of(main_log)))
    ]
    if args.repeat_log:
        rep = protocol
        if args.repeat_limit is not None:
            rep = replace(protocol, time_limit_s=args.repeat_limit)
        repeat_log = Path(args.repeat_log)
        sweeps.append(
            (
                repeat_log,
                rep,
                Path(args.repeat_certificates or certificates_of(repeat_log)),
            )
        )

    checks: list[tuple[Path, ResumeCheck, Path]] = [
        (log, check_resume(log, protocol=proto, config_path=args.config), certs)
        for log, proto, certs in sweeps
    ]
    other = [(log, c, certs) for log, c, certs in checks if c.status == "protocol"]
    outputs = [Path(p) for p in args.outputs]

    if args.discard:
        stale = [(log, c) for log, c, _ in checks if c.status in ("protocol", "mixed")]
        if stale:
            for log, c in stale:
                print(
                    f"{log} was written by another setup and cannot be resumed; "
                    "the smoke test starts afresh:",
                    file=sys.stderr,
                )
                for line in c.problems:
                    print(f"  {line}", file=sys.stderr)
            for p in [x for log, _, certs in checks for x in (log, certs)] + outputs:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
        return 0

    if not other:
        if args.archive:
            print(
                "--archive-previous: no log of another protocol in "
                f"{results_dir}; nothing moved.",
                file=sys.stderr,
            )
        return 0

    # Only the logs of another protocol move, each with its certificates; the
    # encoding times, assets and numbers are derived from the main log and
    # move only with it.  A log of this protocol stays, so its sweep resumes.
    logs = [log for log, _, _ in other]
    derived = [certs for _, _, certs in other]
    if any(log == main_log for log in logs):
        derived += outputs
    kept = [(log, c) for log, c, _ in checks if c.status not in ("protocol", "new")]
    name = archive_name(logs)
    if args.archive:
        try:
            dest = archive(
                results_dir, logs, derived, config=args.config, dest_name=name
            )
        except FileExistsError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(
            f"Moved {', '.join(str(p) for p in logs)} and the files derived from "
            f"{'them' if len(logs) > 1 else 'it'} to {dest}/ (see its README.md).",
            file=sys.stderr,
        )
        for log, c in kept:
            state = "it can be resumed" if c.status == "ok" else "its setup differs"
            print(f"Left {log} in place ({state}).", file=sys.stderr)
        return 0

    print("", file=sys.stderr)
    for log, c, _ in other:
        print(f"{log} holds runs of another protocol:", file=sys.stderr)
        for line in c.problems:
            print(f"  {line}", file=sys.stderr)
    first = name == FIRST_RUN_DIR
    print(
        f"\nThe sweep runs {describe_protocol(protocol)} ({args.config}) and does "
        "not resume a log of another protocol.",
        file=sys.stderr,
    )
    what = "the first run (N = 30, initial play from smt(F))" if first else "that run"
    cmd = "./reproduce.sh --archive-previous" + (f" {args.hint}" if args.hint else "")
    print(
        f"Keep the outputs of {what} by moving them to {results_dir / name}/:\n\n"
        f"    {cmd}              # move, then run the pipeline\n"
        f"    {cmd} --only none  # move only\n\n"
        f"or move {', '.join(str(p) for p in logs)} and the files derived from "
        f"{'them' if len(logs) > 1 else 'it'} yourself, or use another --results-dir.",
        file=sys.stderr,
    )
    for log, c in kept:
        state = "it can be resumed" if c.status == "ok" else "its setup differs"
        print(f"{log} would stay in place ({state}).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
