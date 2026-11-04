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
    time_limit_s = 900               # wall-clock limit per run
    runs = 1                         # runs per (instance, configuration)
    max_iters = "none"               # bound N of IBIS, S-IBIS, Q-IBIS ("none": no bound)
    initial = "collective"           # initial play: "satisfying" smt(F) or "collective" smt(F, f)
    cegar_inner_max_iters = 30       # bound of the inner synthesiser of CEGAR-NE at a coarse partition
    cegar_discrete_max_iters = "none"  # ... and at the discrete partition
    demand = 0                       # d(a) for every agent
    seed = 0                         # seed of the Q-IBIS learners

An iteration bound is a non-negative integer or the string ``"none"`` (TOML
has no null), which means no bound.  A missing CEGAR-NE bound takes the
value of ``max_iters``, and a missing ``initial`` is ``"satisfying"``: a
table without these keys gives the protocol of the first run of the thesis
evaluation, in which every synthesiser, the inner one included, had the
bound ``N = 30`` and started from ``smt(F)``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rcmas.smt_solve import INITIAL_QUERIES

from .scenarios import GRIDS_DIR, Scenario

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomli is a dependency there
    import tomli as tomllib  # type: ignore[no-redef]


NO_BOUND = "none"
"""The TOML spelling of an iteration bound that does not bound."""

SYNTHESIS_FIELDS = (
    "max_iters",
    "initial",
    "cegar_inner_max_iters",
    "cegar_discrete_max_iters",
)
"""The fields of :class:`Protocol` that change what a synthesiser computes."""


@dataclass(frozen=True, slots=True)
class Protocol:
    """Parameters shared by every run of a sweep (thesis Section 7.2).

    ``max_iters`` is the bound ``N`` of the direct runs of IBIS, S-IBIS and
    Q-IBIS, and ``cegar_inner_max_iters`` and ``cegar_discrete_max_iters``
    are the bounds of the inner synthesiser of CEGAR-NE at a coarse and at
    the discrete partition; ``None`` means no bound.  ``initial`` is the
    query for the initial play of every synthesis call.  The defaults are
    the protocol of the first run (``N = 30`` everywhere, ``smt(F)``);
    ``experiments/thesis-eval.toml`` sets the revised protocol.
    """

    time_limit_s: int = 900
    runs: int = 1
    max_iters: int | None = 30
    demand: int = 0
    seed: int = 0
    initial: str = "satisfying"
    cegar_inner_max_iters: int | None = 30
    cegar_discrete_max_iters: int | None = 30

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def synthesis(self) -> dict[str, Any]:
        """The fields that change what a synthesiser computes (:data:`SYNTHESIS_FIELDS`)."""
        return {k: getattr(self, k) for k in SYNTHESIS_FIELDS}


def bound_text(bound: int | None) -> str:
    """An iteration bound as text: ``30`` or ``none``."""
    return NO_BOUND if bound is None else str(bound)


def parse_bound(value: Any, key: str = "bound") -> int | None:
    """An iteration bound from TOML or a command line: an integer >= 0 or ``"none"``."""
    if isinstance(value, str) and value.strip().lower() == NO_BOUND:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"{key} must be an integer >= 0 or {NO_BOUND!r}, got {value!r}"
        )
    try:
        bound = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{key} must be an integer >= 0 or {NO_BOUND!r}, got {value!r}"
        ) from None
    if bound < 0 or (isinstance(value, float) and value != bound):
        raise ValueError(
            f"{key} must be an integer >= 0 or {NO_BOUND!r}, got {value!r}"
        )
    return bound


def normalise_protocol(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """A recorded protocol with the fields that older logs do not record filled in.

    Logs of the first run record only ``time_limit_s``, ``runs``,
    ``max_iters``, ``demand`` and ``seed``.  Their runs started from
    ``smt(F)``, and the inner synthesiser of CEGAR-NE had the bound
    ``max_iters`` at every partition, so the missing fields take these
    values.
    """
    if raw is None:
        return None
    out = dict(raw)
    out.setdefault("initial", "satisfying")
    out.setdefault("cegar_inner_max_iters", out.get("max_iters"))
    out.setdefault("cegar_discrete_max_iters", out.get("max_iters"))
    return out


def protocol_from_table(raw: dict[str, Any], where: str = "[protocol]") -> Protocol:
    """A :class:`Protocol` from a ``[protocol]`` table (see the module docstring)."""
    known = set(Protocol.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown [protocol] keys in {where}: {sorted(unknown)}")
    vals = dict(raw)
    for key in ("max_iters", "cegar_inner_max_iters", "cegar_discrete_max_iters"):
        if key in vals:
            vals[key] = parse_bound(vals[key], key)
    # (With slots, Protocol.max_iters is a descriptor; read the declared default.)
    vals.setdefault("max_iters", Protocol.__dataclass_fields__["max_iters"].default)
    for key in ("cegar_inner_max_iters", "cegar_discrete_max_iters"):
        vals.setdefault(key, vals["max_iters"])
    proto = Protocol(**vals)
    if proto.initial not in INITIAL_QUERIES:
        raise ValueError(
            f"invalid [protocol] in {where}: initial must be one of "
            f"{INITIAL_QUERIES}, got {proto.initial!r}"
        )
    if proto.time_limit_s <= 0 or proto.runs < 1:
        raise ValueError(f"invalid [protocol] in {where}: {proto}")
    return proto


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
    """The ``[protocol]`` table of a configuration (defaults: :class:`Protocol`)."""
    return protocol_from_table(protocol_table(path), str(path))


def protocol_table(path: str | Path) -> dict[str, Any]:
    """The raw ``[protocol]`` table of a configuration (``{}`` if it has none)."""
    return dict(_read(path).get("protocol", {}))


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
