"""Benchmark runner (thesis Sections 7.2 and 7.3).

Each run, one ``(instance, configuration, run number)`` triple, executes in a
fresh Python process started with the ``spawn`` method.  Z3 keeps state in
its global context between queries, so a process that is reused for several
runs makes the result of a run depend on the runs before it; a fresh process
makes every result depend only on the instance, the configuration, the
protocol, the Z3 version and the seed.  ``PYTHONHASHSEED`` is fixed to ``0``
in the children for the same reason.

The parent process enforces the wall-clock limit: a run that has not
returned ``time_limit_s`` seconds after it started is killed and recorded as
a timeout (``t/o``), and so is a run that returns after the limit.  Every
Z3 query also gets the limit as its own timeout, so a query never outlives
its run.  An equilibrium returned within the limit is checked by the
independent certifier (:func:`rcmas.certify.certify`, thesis Section 7.5) in
the same child process, with a separate limit; the certification time is not
part of the run time.  A run counts as solved only if the certifier accepts
its play (field ``solved``).

The output is a JSON-lines log.  Every session (a call of :func:`run_batch`)
first writes a ``{"kind": "session", ...}`` record with the provenance of
:mod:`experiments.provenance` and the protocol, then one ``{"kind": "run",
...}`` record per finished run, and at the end a ``{"kind": "session_end",
...}`` record.  Run records carry the outcome of the tool (``ne``, ``n/c``,
``unk``, ``uns`` or, for CEGAR-NE, ``fail``), ``t/o`` or ``error``, the play
(tool sector indices per round), the rewards and social welfare, the
iterations, the CEGAR-NE partition statistics, the time and the
certificate, plus the session id, the tool commit and the Z3 version.  Grid
paths are relative to the tool root.

A session stopped by Ctrl-C, SIGTERM or SIGHUP kills its running children
and writes its ``session_end`` record (with ``interrupted`` and ``signal``).
A child also ends itself as soon as its parent process is gone, for
example after SIGKILL, so no run outlives its session.

A session holds an exclusive lock on its log, so a second session on the
same log is refused (:class:`LogInUseError`).

With ``resume=True`` the runs already in the log are skipped and new records
are appended.  A resumed session must use the same tool commit, the same uncommitted
changes (none, or the same ``git diff``), the same Z3 version,
configuration file and protocol as the sessions already in the log, unless
``allow_mixed`` is set, so that one log never mixes two versions.  The
synthesis protocol (the iteration bounds and the initial query, recorded in
the session header and in every run record) must always be the same:
``allow_mixed`` does not override it (:class:`ProtocolMismatchError`).
Logs of the first run of the thesis evaluation record only ``max_iters``;
their other fields are read as ``initial = "satisfying"`` and CEGAR-NE
bounds equal to ``max_iters`` (:func:`experiments.config.normalise_protocol`).

A run record also describes the initial play of the run: the time of the
initial query (``initial_query_s``; for CEGAR-NE that of the first inner
call, with one entry per inner call in ``cegar_initial_queries`` and in
``cegar_trace``), the rewards of the initial play in ``M``
(``initial_rewards``, for CEGAR-NE of the lift of the first inner call's
initial play), their sum (``initial_welfare``), whether every reward is
``m`` (``initial_all_m``) and whether the run returned the initial play as
its equilibrium (``initial_was_ne``).  The child reports the initial query
as soon as it is answered, so a run killed at the limit keeps these fields;
``initial_query_s`` is ``null`` on a timeout only if the initial query did
not finish.
"""

from __future__ import annotations

import contextlib
import errno
import json
import multiprocessing
import multiprocessing.connection
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, cast

from .config import SYNTHESIS_FIELDS, Protocol, bound_text, normalise_protocol
from .provenance import collect, relative_to_tool, sha256_file
from .scenarios import Scenario

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

SCHEMA = 2
"""Version of the record format (1 = the examined logs, without ``kind``)."""

GRACE_S = 5.0
"""Seconds the parent waits past a limit before it kills a child."""

STARTUP_TIMEOUT_S = 300.0
"""Seconds a child may take to import the tool and report that it started."""

PARENT_POLL_S = 1.0
"""Seconds between two checks of a child that its parent is still alive."""

ORPHAN_EXIT_CODE = 3
"""Exit code of a child that ends because its parent has gone."""

ALGORITHMS = ("ibis", "sibis", "qibis", "cegar-ibis", "cegar-sibis", "cegar-qibis")


# ---------------------------------------------------------------------------
# Algorithm dispatch (runs in the child)
# ---------------------------------------------------------------------------


def _demands(n: int, demand: int) -> tuple[int, ...] | None:
    """The demand vector passed to the tool; ``None`` (every demand 0) for demand 0."""
    return None if demand == 0 else tuple([demand] * n)


def _play_lists(play: Any) -> list[list[int]] | None:
    if play is None:
        return None
    return [[int(x) for x in profile] for profile in play.actions]


def _initial_fields(
    query_s: float | None,
    rewards: Any,
    horizon: int,
    returned_initial: bool,
    status: str | None = None,
) -> dict[str, Any]:
    """The ``initial_*`` fields of a run record.

    ``initial_query_status`` is the answer to the initial query (``"sat"``,
    ``"unsat"`` or ``"unknown"``, the last also when the query reached the
    Z3 timeout).

    ``initial_rewards`` are rewards in ``M`` (for CEGAR-NE those of the lift
    of the first inner call's initial play), so ``initial_all_m`` says
    whether every agent has reward ``m``, which makes the play an
    equilibrium under any completion (Lemma 3.10.1).
    """
    r = None if rewards is None else [int(x) for x in rewards]
    if status is None and query_s is not None:
        status = "sat" if r is not None else "unknown"
    return {
        "initial_query_s": None if query_s is None else round(float(query_s), 4),
        "initial_query_status": status,
        "initial_rewards": r,
        "initial_welfare": None if r is None else sum(r),
        "initial_all_m": None if r is None else all(x == horizon for x in r),
        "initial_was_ne": bool(returned_initial),
    }


def _run_algorithm(
    territory: Any,
    scenario: Scenario,
    protocol: Protocol,
    timeout_ms: int | None,
    report: Any = None,
) -> dict[str, Any]:
    """Run one synthesiser and return the fields of the run record it determines.

    *report* ``(kind, payload)``, if given, is called as soon as an initial
    query has been answered, so that a run that is killed at the limit
    still records it (``"initial"`` for a direct run, ``"cegar_initial"``
    for each inner call of CEGAR-NE).
    """
    alg = scenario.algorithm
    n = scenario.num_agents
    h = scenario.horizon
    sym = scenario.symmetry
    demands = _demands(n, protocol.demand)

    def send(kind: str, payload: dict[str, Any]) -> None:
        if report is not None:
            report(kind, payload)

    def direct_hook(init: Any) -> None:
        send(
            "initial",
            _initial_fields(init.query_s, init.rewards, h, False, init.reason),
        )

    if alg in ("ibis", "sibis"):
        from rcmas.ibis import solve_ibis, solve_sibis

        solve = solve_ibis if alg == "ibis" else solve_sibis
        res = solve(
            territory=territory,
            num_agents=n,
            horizon=h,
            max_iters=protocol.max_iters,
            initial=protocol.initial,
            symmetry=sym,
            timeout_ms=timeout_ms,
            demands=demands,
            on_initial=direct_hook,
        )
        return {
            "tool_outcome": res.outcome,
            "reason": res.reason,
            "play": _play_lists(res.play),
            "rewards": list(res.payoff_by_agent) if res.payoff_by_agent else None,
            "iterations": res.iterations,
            "solver_checks": sum(it.checks for it in res.trace),
            **_initial_fields(
                res.initial_query_s,
                res.initial_rewards,
                h,
                res.returned_initial,
                _initial_status(res),
            ),
        }

    if alg == "qibis":
        from rcmas.qibis import QibisConfig, solve_qibis

        cfg = QibisConfig(
            max_iters=protocol.max_iters,
            initial=protocol.initial,
            symmetry=sym,
            timeout_ms=timeout_ms,
            demands=demands,
            seed=protocol.seed,
        )
        qres = solve_qibis(
            territory=territory,
            num_agents=n,
            horizon=h,
            cfg=cfg,
            on_initial=direct_hook,
        )
        return {
            "tool_outcome": qres.outcome,
            "reason": qres.reason,
            "play": _play_lists(qres.play),
            "rewards": list(qres.payoff_by_agent) if qres.payoff_by_agent else None,
            "iterations": qres.iterations,
            "solver_checks": sum(it.checks for it in qres.trace),
            **_initial_fields(
                qres.initial_query_s,
                qres.initial_rewards,
                h,
                qres.returned_initial,
                _initial_status(qres),
            ),
        }

    if alg in ("cegar-ibis", "cegar-qibis", "cegar-sibis"):
        from rcmas.cegar import solve_cegar

        calls: list[dict[str, Any]] = []
        first_status: list[str] = []

        def cegar_hook(info: Any) -> None:
            call = _cegar_call(info)
            calls.append(call)
            send("cegar_initial", call)
            if info.iteration == 1:
                send(
                    "initial",
                    _initial_fields(
                        info.query_s, info.lifted_rewards, h, False, info.status
                    ),
                )
                first_status.append(info.status)

        cres = solve_cegar(
            territory=territory,
            num_agents=n,
            horizon=h,
            initial_partition=scenario.partition,
            synthesiser=alg.split("-", 1)[1],
            inner_max_iters=protocol.cegar_inner_max_iters,
            discrete_max_iters=protocol.cegar_discrete_max_iters,
            initial=protocol.initial,
            symmetry=sym,
            timeout_ms=timeout_ms,
            demands=demands,
            seed=protocol.seed,
            on_initial=cegar_hook,
        )
        return {
            "tool_outcome": cres.outcome,
            "reason": cres.reason,
            "play": _play_lists(cres.play),
            "rewards": list(cres.payoff_by_agent) if cres.payoff_by_agent else None,
            "iterations": cres.iterations,
            "inner_iterations": cres.inner_iterations,
            "cegar_initial_blocks": cres.initial_partition_size,
            "cegar_final_blocks": cres.final_partition_size,
            "cegar_at_discrete": bool(cres.at_discrete),
            "cegar_refinements": cres.refinements,
            "cegar_trace": [
                {
                    "blocks": it.blocks,
                    "synthesis": it.synthesis,
                    "check": it.check,
                    "inner_iterations": it.inner_iterations,
                    "refinement": it.refinement,
                    "time_s": round(it.time_s, 3),
                    "initial_query_s": (
                        None
                        if it.initial_query_s is None
                        else round(it.initial_query_s, 4)
                    ),
                    "initial_rewards": (
                        None if it.initial_rewards is None else list(it.initial_rewards)
                    ),
                    "returned_initial": bool(it.returned_initial),
                }
                for it in cres.trace
            ],
            "cegar_initial_queries": calls,
            **_initial_fields(
                cres.initial_query_s,
                cres.initial_rewards,
                h,
                cres.returned_initial,
                first_status[0] if first_status else None,
            ),
        }

    raise ValueError(f"unknown algorithm: {alg}")


def _initial_status(res: Any) -> str | None:
    """The answer to the initial query of a direct run."""
    if res.initial_query_s is None:
        return None
    if res.initial_play is not None:
        return "sat"
    return "unknown" if res.reason == "unknown" else "unsat"


def _cegar_call(info: Any) -> dict[str, Any]:
    """One entry of ``cegar_initial_queries``: the initial query of an inner call."""
    return {
        "iteration": int(info.iteration),
        "blocks": int(info.blocks),
        "status": info.status,
        "initial_query_s": (
            None if info.query_s is None else round(float(info.query_s), 4)
        ),
        "initial_rewards": (
            None if info.abstract_rewards is None else list(info.abstract_rewards)
        ),
    }


def _static_info(territory: Any) -> dict[str, Any]:
    """Sizes of the symmetry groups and of the orbit partition (Section 7.1).

    ``geo_group_order`` is ``|H_geo|`` of the whole territory.
    ``symmetry_group_order`` is the order of the group that the symmetry
    constraints and the orbit partition use: ``H_geo`` on a connected
    territory, the group ``H`` of Theorem 5.3.2 on a disconnected one (for
    example 128 on ``2x2-2x2``, whose ``H_geo`` has order 4).
    ``orbit_blocks`` is the number of blocks of the initial partition of
    CEGAR-NE, :func:`rcmas.abstraction.orbit_partition`.
    """
    from rcmas.abstraction import orbit_partition
    from rcmas.symmetry import component_info, territory_automorphisms

    return {
        "geo_group_order": len(territory_automorphisms(territory)),
        "symmetry_group_order": component_info(territory).group_order,
        "orbit_blocks": len(orbit_partition(territory).blocks),
    }


def _certify(
    territory: Any,
    n: int,
    demand: int,
    play: list[list[int]],
    timeout_s: float,
) -> dict[str, Any]:
    from rcmas.certify import certify

    t0 = time.perf_counter()
    res = certify(
        territory,
        n,
        None,
        [demand] * n,
        play,
        deadline=time.monotonic() + timeout_s,
    )
    if res.accepted:
        status = "accepted"
    elif res.keyword == "deadline":
        status = "unchecked"
    else:
        status = "rejected"
    return {
        "status": status,
        "reason": res.reason,
        "rewards": list(res.rewards),
        "best_responses": list(res.best_responses),
        "states_explored": res.states_explored,
        "time_s": round(time.perf_counter() - t0, 3),
    }


def _exit_when_orphaned(parent_pid: int, interval_s: float = PARENT_POLL_S) -> None:
    """End this process as soon as the process *parent_pid* is no longer its parent.

    A child whose runner is gone (killed, crashed, or stopped by a signal
    other than Ctrl-C) is reparented, and without this check it would go on
    computing until the SIGALRM failsafe, sharing the cores with the runs of
    the next session.  A daemon thread polls the parent id, which works on
    Linux and macOS; Z3 releases the GIL during its queries, so the thread
    also runs while a query is being solved.
    """

    def watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(ORPHAN_EXIT_CODE)
            time.sleep(interval_s)

    threading.Thread(target=watch, name="rcmas-parent-watch", daemon=True).start()


def _child_main(conn: Any, task: dict[str, Any]) -> None:
    """Entry point of a child process: one run, reported over *conn*."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _exit_when_orphaned(int(task.get("parent_pid") or os.getppid()))
    protocol = Protocol(**task["protocol"])  # as_dict() of the parent's protocol
    scenario: Scenario = task["scenario"]
    if hasattr(signal, "alarm"):
        # Failsafe: SIGALRM terminates the child if the parent never kills it.
        signal.alarm(
            int(protocol.time_limit_s + task["cert_timeout_s"] + 2 * STARTUP_TIMEOUT_S)
        )
    try:
        from rcmas.model import Territory

        with open(scenario.grid_path, encoding="utf-8") as f:
            territory = Territory.from_ascii(f)
        static = _static_info(territory)
        # Import the synthesisers before the clock starts.
        import rcmas.cegar  # noqa: F401
        import rcmas.ibis  # noqa: F401
        import rcmas.qibis  # noqa: F401

        conn.send(("started", static))
        t0 = time.perf_counter()
        raw = _run_algorithm(
            territory,
            scenario,
            protocol,
            protocol.time_limit_s * 1000,
            report=lambda kind, payload: conn.send((kind, payload)),
        )
        elapsed = time.perf_counter() - t0
        conn.send(("result", raw, elapsed))
        cert = None
        if (
            task["certify"]
            and raw["tool_outcome"] == "ne"
            and elapsed < protocol.time_limit_s
            and raw["play"] is not None
        ):
            cert = _certify(
                territory,
                scenario.num_agents,
                protocol.demand,
                raw["play"],
                task["cert_timeout_s"],
            )
        conn.send(("cert", cert))
    except BaseException as e:  # noqa: BLE001 - reported to the parent
        try:
            conn.send(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def run_key(record: dict[str, Any]) -> tuple:
    """The key that identifies a run in a log."""
    return (
        record["suite"],
        record["name"],
        record["algorithm"],
        bool(record["symmetry"]),
        record.get("partition", "orbit"),
        int(record["run"]),
    )


def _scenario_key(scenario: Scenario, run: int) -> tuple:
    return (
        scenario.suite,
        scenario.name,
        scenario.algorithm,
        bool(scenario.symmetry),
        scenario.partition,
        run,
    )


def family_of(grid_path: str) -> str:
    """The benchmark family of a grid: the directory below ``grids/``."""
    text = grid_path.replace("\\", "/")
    tail = text.rsplit("grids/", 1)[-1]
    return tail.split("/")[0] if "/" in tail else ""


def _base_record(
    scenario: Scenario,
    run: int,
    protocol: Protocol,
    session: dict[str, Any],
    grid_hashes: dict[str, str],
) -> dict[str, Any]:
    grid_rel = relative_to_tool(scenario.grid_path)
    if scenario.grid_path not in grid_hashes:
        grid_hashes[scenario.grid_path] = sha256_file(scenario.grid_path)
    from rcmas.model import Territory

    with open(scenario.grid_path, encoding="utf-8") as f:
        num_sectors = len(Territory.from_ascii(f))
    return {
        "kind": "run",
        "schema": SCHEMA,
        "suite": scenario.suite,
        "name": scenario.name,
        "grid_path": grid_rel,
        "family": family_of(grid_rel),
        "grid_sha256": grid_hashes[scenario.grid_path],
        "num_sectors": num_sectors,
        "num_agents": scenario.num_agents,
        "horizon": scenario.horizon,
        "demands": [protocol.demand] * scenario.num_agents,
        "algorithm": scenario.algorithm,
        "symmetry": scenario.symmetry,
        "partition": scenario.partition,
        "run": run,
        "time_limit_s": protocol.time_limit_s,
        "max_iters": protocol.max_iters,
        "initial": protocol.initial,
        "cegar_inner_max_iters": protocol.cegar_inner_max_iters,
        "cegar_discrete_max_iters": protocol.cegar_discrete_max_iters,
        "seed": protocol.seed,
    }


_RESULT_DEFAULTS: dict[str, Any] = {
    "outcome": None,
    "tool_outcome": None,
    "reason": None,
    "found_ne": False,
    "solved": False,
    "time_s": None,
    "elapsed_s": None,
    "play": None,
    "rewards": None,
    "welfare": None,
    "iterations": None,
    "inner_iterations": None,
    "solver_checks": None,
    "cegar_initial_blocks": None,
    "cegar_final_blocks": None,
    "cegar_at_discrete": None,
    "cegar_refinements": None,
    "cegar_trace": None,
    "cegar_initial_queries": None,
    "initial_query_s": None,
    "initial_query_status": None,
    "initial_rewards": None,
    "initial_welfare": None,
    "initial_all_m": None,
    "initial_was_ne": False,
    "geo_group_order": None,
    "symmetry_group_order": None,
    "orbit_blocks": None,
    "certificate": None,
    "error": None,
}


def _finish_record(
    base: dict[str, Any],
    protocol: Protocol,
    session: dict[str, Any],
    *,
    static: dict[str, Any] | None,
    raw: dict[str, Any] | None,
    elapsed: float | None,
    cert: dict[str, Any] | None,
    killed: bool = False,
    error: tuple[str, str] | None = None,
    streamed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = dict(base)
    rec.update(_RESULT_DEFAULTS)
    if static:
        rec.update(static)
    if streamed:
        # Reported by the child while it ran: what a killed run still knows.
        rec.update(streamed)
    limit = float(protocol.time_limit_s)
    if error is not None:
        rec["outcome"] = "error"
        rec["reason"] = error[0][:500]
        rec["error"] = error[1][-4000:]
        rec["time_s"] = round(elapsed, 3) if elapsed is not None else None
        rec["elapsed_s"] = rec["time_s"]
    elif killed or raw is None:
        rec["outcome"] = "t/o"
        rec["reason"] = "timeout"
        rec["time_s"] = limit
        rec["elapsed_s"] = round(elapsed, 3) if elapsed is not None else None
    else:
        rec.update(raw)
        rec["elapsed_s"] = round(elapsed, 3) if elapsed is not None else None
        if elapsed is not None and elapsed >= limit:
            rec["outcome"] = "t/o"
            rec["reason"] = "timeout"
            rec["time_s"] = limit
        else:
            rec["outcome"] = raw["tool_outcome"]
            rec["time_s"] = rec["elapsed_s"]
        if rec["rewards"] is not None:
            rec["welfare"] = sum(rec["rewards"])
    rec["found_ne"] = rec["outcome"] == "ne"
    # The run returned its initial play only if it returned an equilibrium.
    rec["initial_was_ne"] = bool(rec["found_ne"] and rec.get("initial_was_ne"))
    if rec["found_ne"] and cert is None:
        cert = {"status": "unchecked", "reason": "not certified in this session"}
    rec["certificate"] = cert
    rec["solved"] = bool(rec["found_ne"] and cert and cert.get("status") == "accepted")
    rec["session"] = session["session"]
    rec["tool_commit"] = session["tool"]["commit"]
    rec["tool_dirty"] = session["tool"]["dirty"]
    rec["z3_version"] = session["z3_version"]
    rec["python_version"] = session["python"]["version"]
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rec


# ---------------------------------------------------------------------------
# Log reading and resume
# ---------------------------------------------------------------------------


def read_log(
    path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Read a result log.

    Returns ``(sessions, runs, valid_bytes)``: the session headers, the run
    records (records without ``kind``, written by the examined version of
    the runner, count as runs), and the length of the prefix of the file
    that consists of complete, readable lines.  A line that was cut off,
    for example by a crash, ends the readable prefix.
    """
    sessions: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    valid = 0
    p = Path(path)
    if not p.exists():
        return sessions, runs, 0
    with p.open("rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        end = data.find(b"\n", pos)
        if end < 0:
            break  # an unterminated last line is incomplete
        line = data[pos:end].strip()
        if line:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break
            kind = rec.get("kind", "run")
            if kind == "session":
                sessions.append(rec)
            elif kind == "run":
                runs.append(rec)
        pos = end + 1
        valid = pos
    return sessions, runs, valid


def _compat_key(header: dict[str, Any]) -> dict[str, Any]:
    tool = header.get("tool", {})
    return {
        "tool_commit": tool.get("commit"),
        # Uncommitted changes: a clean and a dirty tree, or two different
        # dirty trees, of one commit are two versions of the tool.
        "tool_dirty": tool.get("dirty"),
        "tool_diff_sha256": tool.get("diff_sha256"),
        "z3_version": header.get("z3_version"),
        "config_sha256": header.get("config_sha256"),
        "protocol": normalise_protocol(header.get("protocol")),
    }


def _synthesis_protocol(header: dict[str, Any]) -> dict[str, Any] | None:
    proto = normalise_protocol(header.get("protocol"))
    if proto is None:
        return None
    return {k: proto.get(k) for k in SYNTHESIS_FIELDS}


def describe_protocol(protocol: Protocol | dict[str, Any]) -> str:
    """The synthesis protocol in one line, e.g. ``N = none, initial collective, ...``."""
    d = protocol.as_dict() if isinstance(protocol, Protocol) else protocol
    d = normalise_protocol(d) or {}
    return (
        f"N = {bound_text(d.get('max_iters'))}, initial {d.get('initial')}, "
        f"CEGAR-NE inner N = {bound_text(d.get('cegar_inner_max_iters'))} "
        f"(coarse) / {bound_text(d.get('cegar_discrete_max_iters'))} (discrete)"
    )


class MixedLogError(RuntimeError):
    """A resumed session differs from the sessions already in the log."""


class ProtocolMismatchError(MixedLogError):
    """A resumed session has another synthesis protocol than the log.

    The iteration bounds and the initial query change what the synthesisers
    compute, so a log never mixes two of them, even with ``allow_mixed``.
    """


def _check_compatible(
    previous: Sequence[dict[str, Any]], current: dict[str, Any]
) -> list[str]:
    problems = []
    cur = _compat_key(current)
    for header in previous:
        old = _compat_key(header)
        for key, value in cur.items():
            if old[key] != value:
                problems.append(
                    f"session {header.get('session')} has {key}={old[key]!r}, "
                    f"this session {value!r}"
                )
    return sorted(set(problems))


def _protocol_mismatches(
    previous: Sequence[dict[str, Any]], current: dict[str, Any]
) -> list[str]:
    cur = _synthesis_protocol(current)
    out = []
    for header in previous:
        old = _synthesis_protocol(header)
        if old != cur:
            out.append(
                f"session {header.get('session')} ran with "
                f"{describe_protocol(header.get('protocol') or {})}, this session "
                f"with {describe_protocol(current.get('protocol') or {})}"
            )
    return sorted(set(out))


@dataclass(frozen=True)
class ResumeCheck:
    """Whether a session may resume a log (see :func:`check_resume`).

    Attributes:
        status: ``"new"`` (no log, or an empty one), ``"ok"``, ``"protocol"``
            (the log has another synthesis protocol; never resumable) or
            ``"mixed"`` (another commit, Z3 version, configuration or
            protocol field; resumable only with ``allow_mixed``).
        problems: one line per difference.
    """

    status: str
    problems: tuple[str, ...] = ()


def _session_header(
    *,
    protocol: Protocol,
    config_path: str | Path | None,
    argv: list[str] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return collect(
        config_path=config_path,
        argv=argv,
        extra={
            "kind": "session",
            "schema": SCHEMA,
            "protocol": protocol.as_dict(),
            **(extra or {}),
        },
    )


def _resume_status(
    sessions: Sequence[dict[str, Any]], header: dict[str, Any]
) -> ResumeCheck:
    wrong_protocol = _protocol_mismatches(sessions, header)
    if wrong_protocol:
        return ResumeCheck("protocol", tuple(wrong_protocol))
    problems = _check_compatible(sessions, header)
    return ResumeCheck("mixed" if problems else "ok", tuple(problems))


def check_resume(
    log_path: str | Path,
    *,
    protocol: Protocol,
    config_path: str | Path | None = None,
) -> ResumeCheck:
    """Could a session with *protocol* resume *log_path*?  Nothing is run or written.

    ``./reproduce.sh`` calls this before a sweep, so that it can tell the
    author what to do with a log of another protocol before the runner
    refuses it.
    """
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return ResumeCheck("new")
    sessions, runs, _ = read_log(path)
    if not sessions:
        if runs:
            return ResumeCheck(
                "mixed",
                (f"{path} has no session header (a log of the examined runner)",),
            )
        return ResumeCheck("new")
    header = _session_header(protocol=protocol, config_path=config_path, argv=None)
    return _resume_status(sessions, header)


# ---------------------------------------------------------------------------
# Parallel execution with one fresh process per run
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    scenario: Scenario
    run: int
    process: Any
    conn: Any
    deadline: float
    spawned: float
    phase: str = "starting"  # starting -> running -> certifying
    started: float | None = None
    static: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    elapsed: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _order(work: list[tuple[Scenario, int]]) -> list[tuple[Scenario, int]]:
    """Largest instances first, so the long runs do not form the tail of the sweep."""
    from rcmas.model import Territory

    sizes: dict[str, int] = {}
    for sc, _ in work:
        if sc.grid_path not in sizes:
            with open(sc.grid_path, encoding="utf-8") as f:
                sizes[sc.grid_path] = len(Territory.from_ascii(f))
    return sorted(
        work,
        key=lambda item: (
            -sizes[item[0].grid_path],
            -item[0].horizon,
            item[0].name,
            item[1],
        ),
    )


def _kill(job: _Job, *, graceful: bool = False) -> None:
    """Stop the child of *job*; with *graceful*, give it a moment to exit first."""
    if graceful:
        job.process.join(timeout=5)
    if job.process.is_alive():
        try:
            job.process.kill()
        except Exception:  # noqa: BLE001
            pass
    job.process.join(timeout=10)
    try:
        job.conn.close()
    except Exception:  # noqa: BLE001
        pass


def _execute(
    work: list[tuple[Scenario, int]],
    *,
    protocol: Protocol,
    workers: int,
    certify: bool,
    cert_timeout_s: float,
    on_record: Any,
    make_record: Any,
    progress: bool,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    pending = list(work)
    pending.reverse()  # pop() from the end keeps the given order
    jobs: dict[Any, _Job] = {}
    total = len(work)
    done = 0
    limit = float(protocol.time_limit_s)

    def start(scenario: Scenario, run: int) -> None:
        parent, child = ctx.Pipe(duplex=False)
        task = {
            "scenario": scenario,
            "run": run,
            "protocol": protocol.as_dict(),
            "certify": certify,
            "cert_timeout_s": cert_timeout_s,
            "parent_pid": os.getpid(),
        }
        proc = ctx.Process(target=_child_main, args=(child, task), daemon=True)
        proc.start()
        child.close()
        now = time.monotonic()
        jobs[parent] = _Job(
            scenario=scenario,
            run=run,
            process=proc,
            conn=parent,
            deadline=now + STARTUP_TIMEOUT_S,
            spawned=now,
        )

    def finish(job: _Job, record: dict[str, Any]) -> None:
        nonlocal done
        jobs.pop(job.conn, None)
        done += 1
        on_record(record)
        if progress:
            cert = record.get("certificate") or {}
            extra = f", certificate {cert.get('status')}" if record["found_ne"] else ""
            print(
                f"  [{done}/{total}] {record['name']} r{record['run']}: "
                f"{record['outcome']} ({record['time_s'] or 0.0:.1f} s{extra})",
                file=sys.stderr,
                flush=True,
            )

    try:
        while pending or jobs:
            while pending and len(jobs) < workers:
                start(*pending.pop())
            now = time.monotonic()
            wait_s = max(0.0, min(j.deadline for j in jobs.values()) - now)
            ready = multiprocessing.connection.wait(
                list(jobs), timeout=min(wait_s, 60.0)
            )
            for conn in ready:
                job = jobs.get(conn)
                if job is None:
                    continue
                try:
                    msg = cast(Any, conn).recv()
                except (EOFError, OSError):
                    job.process.join(timeout=10)
                    code = job.process.exitcode
                    _kill(job)
                    if job.phase == "certifying":
                        cert = {
                            "status": "unchecked",
                            "reason": f"deadline: certifier process ended (exit code {code})",
                        }
                        finish(job, make_record(job, cert=cert))
                    else:
                        err = (f"worker process ended (exit code {code})", "")
                        finish(job, make_record(job, error=err))
                    continue
                kind = msg[0]
                if kind == "started":
                    job.phase = "running"
                    job.static = msg[1]
                    job.started = time.monotonic()
                    job.deadline = job.started + limit + GRACE_S
                elif kind == "initial":
                    job.extra.update(msg[1])
                elif kind == "cegar_initial":
                    job.extra.setdefault("cegar_initial_queries", []).append(msg[1])
                elif kind == "result":
                    job.raw, job.elapsed = msg[1], msg[2]
                    job.phase = "certifying"
                    job.deadline = time.monotonic() + cert_timeout_s + GRACE_S
                elif kind == "cert":
                    _kill(job, graceful=True)
                    finish(job, make_record(job, cert=msg[1]))
                elif kind == "error" and job.phase == "certifying":
                    _kill(job, graceful=True)
                    cert = {"status": "unchecked", "reason": f"error: {msg[1]}"}
                    finish(job, make_record(job, cert=cert))
                elif kind == "error":
                    _kill(job, graceful=True)
                    elapsed = (
                        time.monotonic() - job.started
                        if job.started is not None
                        else None
                    )
                    job.elapsed = elapsed
                    finish(job, make_record(job, error=(msg[1], msg[2])))
            now = time.monotonic()
            for job in [j for j in jobs.values() if now >= j.deadline]:
                _kill(job)
                if job.phase == "starting":
                    finish(job, make_record(job, error=("worker did not start", "")))
                elif job.phase == "running":
                    job.elapsed = now - (job.started or now)
                    finish(job, make_record(job, killed=True))
                else:
                    cert = {
                        "status": "unchecked",
                        "reason": "deadline: certifier killed",
                    }
                    finish(job, make_record(job, cert=cert))
    except BaseException:
        for job in list(jobs.values()):
            _kill(job)
        raise


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


class LogInUseError(RuntimeError):
    """Another session is writing the log."""


@contextlib.contextmanager
def _exclusive_log(path: Path | None) -> Iterator[None]:
    """Hold an exclusive lock on the log *path* while the block runs.

    Two sessions on one log would both run every remaining run, at twice the
    load, and could cut each other's lines when they repair the tail of the
    log.  The lock is an ``flock`` on the log file itself, so it needs no
    extra file and ends with the process that holds it, however that
    process ends.  A log that this call creates and the session leaves
    empty is removed.  Without ``fcntl`` (Windows) there is no lock.
    """
    if path is None or fcntl is None:
        yield
        return
    existed = path.exists()
    if not existed:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise
            raise LogInUseError(
                f"another session is writing {path}; let it finish or stop it "
                "(Ctrl-C) before you resume"
            ) from None
        try:
            yield
        finally:
            if not existed and os.fstat(fd).st_size == 0:
                with contextlib.suppress(OSError):
                    os.unlink(path)
    finally:
        os.close(fd)


class SessionTerminated(KeyboardInterrupt):
    """The session was stopped by SIGTERM or SIGHUP.

    A subclass of :class:`KeyboardInterrupt`, so that a session stopped by
    ``kill`` (or by a closed terminal) is wound down like one stopped by
    Ctrl-C: the running children are killed and the log gets its
    ``session_end`` record.
    """

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        self.signame = signal.Signals(signum).name
        super().__init__(self.signame)


@contextlib.contextmanager
def _stop_on_signals() -> Iterator[None]:
    """Raise :class:`SessionTerminated` on SIGTERM and SIGHUP while the block runs.

    Only the main thread can set signal handlers; elsewhere this does
    nothing.  Repeated signals during the cleanup are ignored, so that the
    ``session_end`` record is written in full.  The previous handlers are
    restored afterwards.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signums = [
        getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
    ]
    previous: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        for s in signums:
            signal.signal(s, signal.SIG_IGN)
        raise SessionTerminated(signum)

    for s in signums:
        previous[s] = signal.signal(s, handler)
    try:
        yield
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


def run_batch(
    scenarios: Iterable[Scenario],
    *,
    protocol: Protocol = Protocol(),
    output_path: str | Path | None = None,
    workers: int = 1,
    resume: bool = False,
    progress: bool = True,
    certify: bool = True,
    cert_timeout_s: float = 1800.0,
    config_path: str | Path | None = None,
    argv: list[str] | None = None,
    allow_mixed: bool = False,
    order: bool = True,
) -> list[dict[str, Any]]:
    """Run every scenario ``protocol.runs`` times and return the run records.

    Records are appended to *output_path* as they finish.  With *resume*,
    the runs already in the log are skipped; the returned list then also
    holds the earlier records.  Raises :class:`MixedLogError` if a resumed
    log was written with another tool commit or other uncommitted changes,
    Z3 version, configuration file or protocol (unless *allow_mixed*), and :class:`LogInUseError` if another
    session is writing *output_path*.
    """
    scenarios = list(scenarios)
    for sc in scenarios:
        if sc.algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm: {sc.algorithm}")
    workers = max(1, int(workers))
    os.environ["PYTHONHASHSEED"] = "0"  # inherited by the spawned children
    out_path = Path(output_path) if output_path is not None else None
    with _exclusive_log(out_path):
        return _run_session(
            scenarios,
            protocol=protocol,
            out_path=out_path,
            workers=workers,
            resume=resume,
            progress=progress,
            certify=certify,
            cert_timeout_s=cert_timeout_s,
            config_path=config_path,
            argv=argv,
            allow_mixed=allow_mixed,
            order=order,
        )


def _run_session(
    scenarios: list[Scenario],
    *,
    protocol: Protocol,
    out_path: Path | None,
    workers: int,
    resume: bool,
    progress: bool,
    certify: bool,
    cert_timeout_s: float,
    config_path: str | Path | None,
    argv: list[str] | None,
    allow_mixed: bool,
    order: bool,
) -> list[dict[str, Any]]:
    """The body of :func:`run_batch`, run while this session holds the log."""
    session_id = uuid.uuid4().hex[:12]
    header = _session_header(
        protocol=protocol,
        config_path=config_path,
        argv=argv,
        extra={
            "session": session_id,
            "workers": workers,
            "certify": certify,
            "cert_timeout_s": cert_timeout_s,
            "scenarios": len(scenarios),
        },
    )

    prior: list[dict[str, Any]] = []
    done_keys: set[tuple] = set()
    if out_path is not None and resume and out_path.exists():
        sessions, prior, valid = read_log(out_path)
        check = _resume_status(sessions, header)
        if check.status == "protocol":
            raise ProtocolMismatchError(
                f"{out_path} was written with another synthesis protocol (iteration "
                "bounds or initial query); resume would mix two protocols in one log, "
                "which --allow-mixed does not permit. Move the log away or use a new "
                "output file.\n  " + "\n  ".join(check.problems)
            )
        if check.problems and not allow_mixed:
            raise MixedLogError(
                f"{out_path} was written by a different setup; resume would mix "
                "results. Use a new output file, or pass --allow-mixed.\n  "
                + "\n  ".join(check.problems)
            )
        if valid < out_path.stat().st_size:
            with open(out_path, "r+b") as f:
                f.truncate(valid)
            if progress:
                print(
                    f"Removed an incomplete last line from {out_path}.", file=sys.stderr
                )
        done_keys = {run_key(r) for r in prior}
    elif (
        out_path is not None
        and out_path.exists()
        and out_path.stat().st_size
        and not resume
    ):
        raise FileExistsError(
            f"{out_path} exists; pass --resume to continue it or choose another file"
        )

    work = [
        (sc, run)
        for run in range(1, protocol.runs + 1)
        for sc in scenarios
        if _scenario_key(sc, run) not in done_keys
    ]
    if order:
        work = _order(work)
    skipped = len(scenarios) * protocol.runs - len(work)
    if progress:
        tool = header["tool"]
        print(
            f"rcmas {tool['version']} at {str(tool['commit'])[:12]}"
            f"{' (dirty)' if tool['dirty'] else ''}, Python {header['python']['version']}, "
            f"z3-solver {header['z3_version']}, {header['machine']['cpu_model']}",
            file=sys.stderr,
        )
        print(
            f"{len(work)} runs to do ({skipped} already in the log), {workers} workers, "
            f"limit {protocol.time_limit_s} s, {describe_protocol(protocol)}",
            file=sys.stderr,
        )
        if tool["dirty"]:
            print(
                "WARNING: the tool has uncommitted changes; the log records tool_dirty=true.",
                file=sys.stderr,
            )

    records: list[dict[str, Any]] = []
    if not work:
        return prior
    out = open(out_path, "a", encoding="utf-8") if out_path is not None else None
    grid_hashes: dict[str, str] = {}

    def write(rec: dict[str, Any]) -> None:
        if out is not None:
            out.write(json.dumps(rec) + "\n")
            out.flush()

    def on_record(rec: dict[str, Any]) -> None:
        records.append(rec)
        write(rec)

    def make_record(job: _Job, **kw: Any) -> dict[str, Any]:
        base = _base_record(job.scenario, job.run, protocol, header, grid_hashes)
        return _finish_record(
            base,
            protocol,
            header,
            static=job.static,
            raw=job.raw if not kw.get("killed") else None,
            elapsed=job.elapsed,
            cert=kw.get("cert"),
            killed=kw.get("killed", False),
            error=kw.get("error"),
            streamed=job.extra,
        )

    interrupted = False
    stopped_by: str | None = None
    with _stop_on_signals():
        try:
            write(header)
            _execute(
                work,
                protocol=protocol,
                workers=workers,
                certify=certify,
                cert_timeout_s=cert_timeout_s,
                on_record=on_record,
                make_record=make_record,
                progress=progress,
            )
        except KeyboardInterrupt as e:
            interrupted = True
            stopped_by = e.signame if isinstance(e, SessionTerminated) else "SIGINT"
            raise
        finally:
            write(
                {
                    "kind": "session_end",
                    "session": session_id,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "runs_written": len(records),
                    "interrupted": interrupted,
                    "signal": stopped_by,
                }
            )
            if out is not None:
                out.close()
    return prior + records
