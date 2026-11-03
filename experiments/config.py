"""TOML-driven experiment configuration.

A configuration file names the instances, the configurations to run on each
of them, and the protocol of the sweep.  :func:`load_config` returns the
scenarios (one per instance and configuration), :func:`load_protocol` the
protocol.

Instances are given either as a cross product of grids and agent counts,
keeping every number of agents that divides the number of sectors (the
benchmark of thesis Section 7.2)::

    [grids]
    symmetric = ["3x2.txt", "4x4.txt"]

    [parameters]
    agents = [2, 3, 4, 5, 6]

or as an explicit list, which may use any number of agents::

    [[instances]]
    grid = "symmetric/3x2.txt"
    agents = [2, 3]

Both forms can be combined.  The configurations are the cross product of
``algorithms``, ``symmetry`` and, for CEGAR-NE (``cegar-*``), ``partitions``::

    [parameters]
    algorithms = ["ibis", "sibis", "qibis", "cegar-sibis"]
    partitions = ["orbit"]
    symmetry = [false, true]

The optional ``[protocol]`` table fixes the parameters of the runs; the
runner uses them unless its command line overrides them, and records the
values it used::

    [protocol]
    time_limit_s = 900   # wall-clock limit per run
    runs = 1             # runs per (instance, configuration)
    max_iters = 30       # iteration bound N (also of the inner S-IBIS of CEGAR-NE)
    demand = 0           # d(a) for every agent
    seed = 0             # seed of the Q-IBIS learners
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .scenarios import GRIDS_DIR, Scenario

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomli is a dependency there
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True, slots=True)
class Protocol:
    """Parameters shared by every run of a sweep (thesis Section 7.2)."""

    time_limit_s: int = 900
    runs: int = 1
    max_iters: int = 30
    demand: int = 0
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_grids(grids_section: dict[str, list[str]]) -> list[str]:
    """Resolve grid paths from a ``[grids]`` section."""
    paths: list[str] = []
    for category, names in grids_section.items():
        for name in names:
            p = os.path.normpath(os.path.join(GRIDS_DIR, category, name))
            if not os.path.isfile(p):
                raise FileNotFoundError(f"grid not found: {p}")
            paths.append(p)
    return paths


def _num_sectors(grid_path: str) -> int:
    from rcmas.model import Territory

    with open(grid_path, encoding="utf-8") as f:
        return len(Territory.from_ascii(f))


def _feasible_horizon(grid_path: str, num_agents: int) -> int | None:
    """Horizon ``m = |T| / n``, or ``None`` if ``n`` does not divide ``|T|``."""
    S = _num_sectors(grid_path)
    if num_agents < 1 or S % num_agents != 0 or S // num_agents < 1:
        return None
    return S // num_agents


def _read(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_protocol(path: str | Path) -> Protocol:
    """The ``[protocol]`` table of a configuration (defaults: thesis Section 7.2)."""
    raw = _read(path).get("protocol", {})
    known = set(Protocol.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown [protocol] keys in {path}: {sorted(unknown)}")
    proto = Protocol(**raw)
    if proto.time_limit_s <= 0 or proto.runs < 1 or proto.max_iters < 0:
        raise ValueError(f"invalid [protocol] in {path}: {proto}")
    return proto


def _instances(cfg: dict[str, Any]) -> list[tuple[str, int, int]]:
    """``(grid_path, n, m)`` for every instance of the configuration."""
    params = cfg.get("parameters", {})
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int]] = set()
    agents_list: list[int] = params.get("agents", [2])
    for grid_path in _resolve_grids(cfg.get("grids", {})):
        for n in agents_list:
            h = _feasible_horizon(grid_path, n)
            if h is None or (grid_path, n) in seen:
                continue
            seen.add((grid_path, n))
            out.append((grid_path, n, h))
    for entry in cfg.get("instances", []):
        grid_path = os.path.normpath(os.path.join(GRIDS_DIR, entry["grid"]))
        if not os.path.isfile(grid_path):
            raise FileNotFoundError(f"grid not found: {grid_path}")
        S = _num_sectors(grid_path)
        for n in entry.get("agents", [2]):
            if (grid_path, n) in seen:
                continue
            if n < 1 or S // n < 0:
                raise ValueError(f"invalid number of agents {n} for {entry['grid']}")
            seen.add((grid_path, n))
            out.append((grid_path, n, S // n))
    return out


def load_config(path: str | Path) -> list[Scenario]:
    """Load a TOML config and return the cross-product of scenarios.

    Non-CEGAR algorithms (those not starting with ``cegar-``) get a single
    partition entry (``"discrete"``) regardless of the config's partition list,
    to avoid redundant runs.
    """
    cfg = _read(path)
    suite_name = cfg.get("suite", {}).get("name", Path(path).stem)
    params = cfg.get("parameters", {})
    algorithms: list[str] = params.get("algorithms", ["sibis"])
    partitions: list[str] = params.get("partitions", ["discrete"])
    symmetry_list: list[bool] = params.get("symmetry", [False])

    scenarios: list[Scenario] = []
    for grid_path, n, h in _instances(cfg):
        grid_name = os.path.splitext(os.path.basename(grid_path))[0]
        for alg in algorithms:
            is_cegar = alg.startswith("cegar-")
            alg_partitions = partitions if is_cegar else ["discrete"]
            for part in alg_partitions:
                for sym in symmetry_list:
                    name = f"{grid_name}_n{n}_{alg}"
                    if is_cegar:
                        name += f"_{part}"
                    if sym:
                        name += "_sym"
                    scenarios.append(
                        Scenario(
                            suite=suite_name,
                            name=name,
                            grid_path=grid_path,
                            num_agents=n,
                            horizon=h,
                            algorithm=alg,
                            symmetry=sym,
                            partition=part,
                        )
                    )
    names = [s.name for s in scenarios]
    if len(names) != len(set(names)):
        dup = sorted({x for x in names if names.count(x) > 1})
        raise ValueError(f"duplicate scenario names in {path}: {dup[:5]}")
    return scenarios
