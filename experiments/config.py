"""TOML-driven experiment configuration.

Loads a TOML config file and generates :class:`Scenario` objects from the
cross-product of grids, agent counts, algorithms, partitions, and symmetry
flags.  Non-CEGAR algorithms ignore the partition field.

Example config (experiments/cegar.toml)::

    [suite]
    name = "cegar"

    [grids]
    symmetric = ["4x3.txt", "4x4.txt", "6x4.txt"]
    asymmetric = ["interesting.txt"]

    [parameters]
    agents = [2, 3]
    algorithms = ["sibis", "cegar-sibis"]
    partitions = ["discrete", "grid-2x2", "bfs-4"]
    symmetry = [false]
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from .scenarios import GRIDS_DIR, Scenario


def _resolve_grids(grids_section: dict[str, list[str]]) -> list[str]:
    """Resolve grid paths from a ``[grids]`` section."""
    paths: list[str] = []
    for category, names in grids_section.items():
        for name in names:
            p = os.path.join(GRIDS_DIR, category, name)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"grid not found: {p}")
            paths.append(p)
    return paths


def _feasible_horizon(grid_path: str, num_agents: int) -> int | None:
    """Compute horizon = S // n, or None if n doesn't divide S."""
    from rcmas.model import Territory

    with open(grid_path) as f:
        t = Territory.from_ascii(f)
    S = len(t)
    if S % num_agents != 0 or S // num_agents < 1:
        return None
    return S // num_agents


def load_config(path: str | Path) -> list[Scenario]:
    """Load a TOML config and return the cross-product of scenarios.

    Non-CEGAR algorithms (those not starting with ``cegar-``) get a single
    partition entry (``"discrete"``) regardless of the config's partition list,
    to avoid redundant runs.
    """
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    suite_name = cfg.get("suite", {}).get("name", Path(path).stem)
    grid_paths = _resolve_grids(cfg.get("grids", {}))

    params = cfg.get("parameters", {})
    agents_list: list[int] = params.get("agents", [2])
    algorithms: list[str] = params.get("algorithms", ["sibis"])
    partitions: list[str] = params.get("partitions", ["discrete"])
    symmetry_list: list[bool] = params.get("symmetry", [False])

    scenarios: list[Scenario] = []
    for grid_path in grid_paths:
        grid_name = os.path.splitext(os.path.basename(grid_path))[0]
        for n in agents_list:
            h = _feasible_horizon(grid_path, n)
            if h is None:
                continue
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
                        scenarios.append(Scenario(
                            suite=suite_name,
                            name=name,
                            grid_path=grid_path,
                            num_agents=n,
                            horizon=h,
                            algorithm=alg,
                            symmetry=sym,
                            partition=part,
                        ))
    return scenarios
