"""Loading result logs for the thesis assets and numbers (thesis Chapter 7).

:func:`load` reads a JSON-lines log of :mod:`experiments.runner` and returns
a :class:`Results` with one :class:`Run` per (instance, configuration), the
session headers, and the problems found while reading (duplicate runs,
missing certificates, rejected equilibria).  Logs written by the examined
version of the runner (no ``kind`` field) are read too; their outcomes are
derived from ``status``, ``found_ne`` and ``reason``, and they carry no
certificate.

A run is *solved* if it returned an equilibrium within the time limit and
every certificate of it accepts the play: the certificate the runner
attached, and the entry of the re-certification file written by
``rcmas certify --results LOG --output CERTS`` if one is given.  Logs without
certificates can be read with ``require_certificate=False``, which counts
every returned equilibrium as solved (only for the examined logs).
"""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ALGS = ("ibis", "sibis", "qibis", "cegar-sibis")
"""The algorithms of the thesis evaluation, in table order."""

ALG_LABEL = {
    "ibis": "IBIS",
    "sibis": "S-IBIS",
    "qibis": "Q-IBIS",
    "cegar-sibis": "CEGAR",
    "cegar-ibis": "CEGAR(IBIS)",
    "cegar-qibis": "CEGAR(Q-IBIS)",
}
FAMILIES = ("symmetric", "local_symmetry", "disconnected", "asymmetric")
FAMILY_LABEL = {
    "symmetric": "symmetric",
    "local_symmetry": "L-shaped",
    "disconnected": "disconnected",
    "asymmetric": "random",
}
OUTCOMES = ("ne", "t/o", "n/c", "unk", "fail", "uns", "error")


@dataclass(frozen=True)
class Run:
    """One run, normalised over both log formats."""

    grid: str
    grid_path: str
    family: str
    sectors: int
    agents: int
    horizon: int
    algorithm: str
    symmetry: bool
    run: int
    outcome: str
    reason: str | None
    time_s: float
    time_limit_s: float
    found_ne: bool
    certificate: str | None
    solved: bool
    rewards: tuple[int, ...] | None
    play: tuple[tuple[int, ...], ...] | None
    iterations: int | None
    inner_iterations: int | None
    cegar_final_blocks: int | None
    cegar_at_discrete: bool | None
    orbit_blocks: int | None
    name: str
    raw: dict[str, Any] = field(repr=False, compare=False, hash=False)

    @property
    def instance(self) -> tuple[str, int]:
        return (self.grid_path, self.agents)

    @property
    def config(self) -> tuple[str, bool]:
        return (self.algorithm, self.symmetry)

    @property
    def welfare(self) -> int | None:
        return None if self.rewards is None else sum(self.rewards)

    @property
    def all_at_bound(self) -> bool:
        """Every agent has reward ``m`` (the bound of Lemma 3.10.1)."""
        return self.rewards is not None and all(r == self.horizon for r in self.rewards)


def _norm_grid_path(raw: str) -> str:
    text = str(raw).replace("\\", "/")
    if "grids/" in text:
        return "grids/" + text.rsplit("grids/", 1)[1]
    return text


def _family(grid_path: str) -> str:
    tail = grid_path.rsplit("grids/", 1)[-1]
    return tail.split("/")[0] if "/" in tail else ""


def _legacy_outcome(d: dict[str, Any]) -> str:
    if d.get("status") == "timeout":
        return "t/o"
    if d.get("status") == "error":
        return "error"
    if d.get("found_ne"):
        return "ne"
    reason = str(d.get("reason"))
    return {"max_iters": "n/c", "cycle": "n/c", "unsat": "uns", "unknown": "unk"}.get(
        reason, reason
    )


@dataclass
class Results:
    """The runs of one log, indexed by instance and configuration."""

    runs: list[Run]
    sessions: list[dict[str, Any]]
    problems: list[str]
    path: str
    certificates_path: str | None = None
    recertified: int = 0

    def __post_init__(self) -> None:
        self.index: dict[tuple[tuple[str, int], tuple[str, bool]], Run] = {
            (r.instance, r.config): r for r in self.runs
        }

    # -- basic views ----------------------------------------------------
    @property
    def instances(self) -> list[tuple[str, int]]:
        """Instances ordered by number of sectors, grid name, number of agents."""
        info = {r.instance: (r.sectors, r.grid, r.agents) for r in self.runs}
        return sorted(info, key=lambda k: info[k])

    def instance_info(self, inst: tuple[str, int]) -> Run:
        for r in self.runs:
            if r.instance == inst:
                return r
        raise KeyError(inst)

    @property
    def configs(self) -> list[tuple[str, bool]]:
        present = {r.config for r in self.runs}
        ordered = [(a, s) for a in ALGS for s in (False, True) if (a, s) in present]
        extra = sorted(present - set(ordered))
        return ordered + extra

    @property
    def algorithms(self) -> list[str]:
        seen: list[str] = []
        for a, _ in self.configs:
            if a not in seen:
                seen.append(a)
        return seen

    def get(self, inst: tuple[str, int], alg: str, sym: bool) -> Run | None:
        return self.index.get((inst, (alg, sym)))

    def runs_of(self, alg: str, sym: bool) -> list[Run]:
        return [r for r in self.runs if r.config == (alg, sym)]

    def solved_of(self, alg: str, sym: bool) -> list[Run]:
        return [r for r in self.runs_of(alg, sym) if r.solved]

    def solved_instances(self, alg: str, sym: bool) -> set[tuple[str, int]]:
        return {r.instance for r in self.solved_of(alg, sym)}

    def median_time(self, alg: str, sym: bool) -> float | None:
        ts = [r.time_s for r in self.solved_of(alg, sym)]
        return statistics.median(ts) if ts else None

    def largest_solved(self, alg: str, sym: bool) -> Run | None:
        ok = self.solved_of(alg, sym)
        return max(ok, key=lambda r: (r.sectors, r.agents)) if ok else None

    @property
    def time_limit_s(self) -> float:
        limits = {r.time_limit_s for r in self.runs}
        return max(limits) if limits else 900.0


def _read_certificates(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[(str(row.get("name")), int(row.get("run") or 1))] = row
    return out


def load(
    path: str | Path,
    *,
    certificates: str | Path | None = None,
    require_certificate: bool = True,
    run_number: int | None = 1,
) -> Results:
    """Read a result log (see the module docstring).

    *run_number* keeps only that run of every (instance, configuration);
    ``None`` keeps all of them (the last record of a key wins).
    """
    sessions: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    problems: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{path}:{lineno}: unreadable line skipped")
                continue
            kind = d.get("kind", "run")
            if kind == "session":
                sessions.append(d)
            elif kind == "run":
                raw_runs.append(d)

    # The examined logs do not record the limit; a timeout records it as its time.
    legacy_limit = max(
        (float(d.get("time_s") or 0) for d in raw_runs if d.get("status") == "timeout"),
        default=900.0,
    )
    certs = _read_certificates(certificates) if certificates else {}
    recertified = 0
    by_key: dict[tuple, Run] = {}
    for d in raw_runs:
        if run_number is not None and int(d.get("run", 1)) != run_number:
            continue
        grid_path = _norm_grid_path(d["grid_path"])
        legacy = "outcome" not in d
        outcome = _legacy_outcome(d) if legacy else str(d["outcome"])
        found_ne = outcome == "ne"
        statuses: list[str] = []
        inline = d.get("certificate")
        if isinstance(inline, dict) and inline.get("status"):
            statuses.append(str(inline["status"]))
        ext = certs.get((str(d.get("name")), int(d.get("run", 1))))
        if ext is not None and found_ne:
            statuses.append(str(ext.get("status")))
            recertified += 1
        if not found_ne:
            cert = None
            solved = False
        elif statuses:
            if "rejected" in statuses:
                cert = "rejected"
            elif all(s == "accepted" for s in statuses):
                cert = "accepted"
            else:
                cert = "unchecked"
            if len(set(statuses)) > 1:
                problems.append(
                    f"{d.get('name')}: certificates disagree ({', '.join(statuses)})"
                )
            solved = cert == "accepted"
        else:
            cert = None
            solved = not require_certificate
            if require_certificate:
                problems.append(f"{d.get('name')}: equilibrium without a certificate")
        if found_ne and cert == "rejected":
            problems.append(f"{d.get('name')}: the certifier REJECTED the equilibrium")
        if found_ne and cert == "unchecked":
            problems.append(f"{d.get('name')}: the certifier did not finish")
        play = d.get("play")
        rewards = d.get("rewards", d.get("payoff"))
        run = Run(
            grid=os.path.splitext(os.path.basename(grid_path))[0],
            grid_path=grid_path,
            family=d.get("family") or _family(grid_path),
            sectors=int(d["num_sectors"]),
            agents=int(d["num_agents"]),
            horizon=int(d["horizon"]),
            algorithm=str(d["algorithm"]),
            symmetry=bool(d["symmetry"]),
            run=int(d.get("run", 1)),
            outcome=outcome,
            reason=d.get("reason"),
            time_s=float(d.get("time_s") or 0.0),
            time_limit_s=float(d.get("time_limit_s") or legacy_limit),
            found_ne=found_ne,
            certificate=cert,
            solved=solved,
            rewards=tuple(rewards) if rewards else None,
            play=tuple(tuple(p) for p in play) if play else None,
            iterations=d.get("iterations"),
            inner_iterations=d.get("inner_iterations"),
            cegar_final_blocks=d.get("cegar_final_blocks"),
            cegar_at_discrete=d.get("cegar_at_discrete"),
            orbit_blocks=d.get("orbit_blocks"),
            name=str(d.get("name")),
            raw=d,
        )
        key = (run.instance, run.config, run.run)
        if key in by_key:
            problems.append(
                f"{run.name} r{run.run}: duplicate record, the last one is used"
            )
        by_key[key] = run
    return Results(
        runs=list(by_key.values()),
        sessions=sessions,
        problems=problems,
        path=str(path),
        certificates_path=str(certificates) if certificates else None,
        recertified=recertified,
    )


def config_label(alg: str, sym: bool) -> str:
    return ALG_LABEL.get(alg, alg) + ("+sym" if sym else "")


def group_by(items: Iterable[Run], key: Any) -> dict[Any, list[Run]]:
    out: dict[Any, list[Run]] = defaultdict(list)
    for r in items:
        out[key(r)].append(r)
    return out
