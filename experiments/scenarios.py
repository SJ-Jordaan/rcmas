"""Experiment scenario definitions.

Each scenario is a (grid, agents, horizon, algorithm, flags) tuple with metadata
explaining *why* it's included — which feature it tests, what the control is, and
what we expect to see.

Scenario suites:
    E1 — Global symmetry breaking (rectangular grids, D4/D2 Aut)
    E2 — Local symmetry breaking (L-shapes, trivial global Aut)
    E3 — Disconnected territory handling (multi-component grids)
    E4 — CEGAR vs direct synthesis (orbit/discrete partition vs IBIS/Q-IBIS)
    E5 — Q-IBIS as CEGAR synthesiser (CEGAR+IBIS vs CEGAR+Q-IBIS)
    E6 — Random scenarios (procedurally generated grids)
    E7 — Scalability (increasing grid sizes)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

GRIDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grids")


@dataclass(frozen=True, slots=True)
class Scenario:
    """A single benchmark scenario."""

    suite: str
    name: str
    grid_path: str
    num_agents: int
    horizon: int
    algorithm: str  # "ibis", "qibis", "cegar-ibis", "cegar-qibis"
    symmetry: bool
    partition: str = "orbit"  # only for cegar algorithms
    demands: tuple[int, ...] | None = None
    description: str = ""


def _grid(category: str, name: str) -> str:
    return os.path.join(GRIDS_DIR, category, name)


def _feasible_horizons(S: int, n: int) -> int:
    """Horizon = S / n.  Requires S divisible by n (for require_victory)."""
    if S % n != 0:
        raise ValueError(f"S={S} not divisible by n={n}")
    return S // n


# ---------------------------------------------------------------------------
# E1: Global symmetry breaking
# ---------------------------------------------------------------------------

def e1_global_symmetry() -> list[Scenario]:
    """Rectangular grids with D4/D2 automorphisms. Compare sym on/off.

    These grids have non-trivial Aut(T):
    - Squares (NxN): |Aut| = 8 (D4)
    - Rectangles (NxM, N!=M): |Aut| = 4 (V4)

    Symmetry breaking should prune equivalent search branches.
    """
    scenarios: list[Scenario] = []
    configs = [
        # (grid_file, S, agent_counts)
        ("3x3.txt", 9, [3]),
        ("4x4.txt", 16, [2, 4]),
        ("4x3.txt", 12, [2, 3, 4]),
        ("5x4.txt", 20, [2, 4, 5]),
        ("6x4.txt", 24, [2, 3, 4]),
        ("5x5.txt", 25, [5]),
        ("6x6.txt", 36, [2, 3, 4, 6]),
    ]

    for grid_file, S, agent_counts in configs:
        path = _grid("symmetric", grid_file)
        for n in agent_counts:
            h = _feasible_horizons(S, n)
            for sym in [False, True]:
                for alg in ["ibis", "qibis"]:
                    label = f"{grid_file[:-4]}_n{n}_{alg}_sym{sym}"
                    scenarios.append(Scenario(
                        suite="E1", name=label, grid_path=path,
                        num_agents=n, horizon=h, algorithm=alg,
                        symmetry=sym,
                        description=f"Global symmetry: {grid_file} with {n} agents, sym={sym}",
                    ))
    return scenarios


# ---------------------------------------------------------------------------
# E2: Local symmetry breaking
# ---------------------------------------------------------------------------

def e2_local_symmetry() -> list[Scenario]:
    """L-shaped/irregular grids with trivial global Aut but non-trivial local
    subregion automorphisms.

    Local symmetry breaking activates only when:
    1. |Aut(T)| = 1 (trivial global)
    2. A rectangular subregion R has |Aut(R)| > 1
    3. Boundary pairs exist between R and T\\R

    Symmetry=True enables both global (no-op here) and local constraints.
    """
    scenarios: list[Scenario] = []
    configs = [
        ("L-4x4-2.txt", 14, [2, 7]),
        ("L-5x5-4.txt", 22, [2, 11]),
        ("L-6x6-6.txt", 30, [2, 3, 5]),
        ("L-6x4-4.txt", 16, [2, 4, 8]),
        # T-5x5.txt has S=13 (prime) — no useful divisor for n>=2 with h>=2
        ("L-8x6-12.txt", 36, [2, 3, 4, 6]),
    ]

    for grid_file, S, agent_counts in configs:
        path = _grid("local_symmetry", grid_file)
        for n in agent_counts:
            h = _feasible_horizons(S, n)
            if h < 1:
                continue
            for sym in [False, True]:
                for alg in ["ibis", "qibis"]:
                    label = f"{grid_file[:-4]}_n{n}_{alg}_sym{sym}"
                    scenarios.append(Scenario(
                        suite="E2", name=label, grid_path=path,
                        num_agents=n, horizon=h, algorithm=alg,
                        symmetry=sym,
                        description=f"Local symmetry: {grid_file} with {n} agents, sym={sym}",
                    ))
    return scenarios


# ---------------------------------------------------------------------------
# E3: Disconnected territory handling
# ---------------------------------------------------------------------------

def e3_disconnected() -> list[Scenario]:
    """Disconnected grids — multiple connected components.

    Isomorphic components enable wreath product symmetry (inter-component
    ordering). Non-isomorphic components only use intra-component automorphisms.
    """
    scenarios: list[Scenario] = []
    configs = [
        # (grid_file, S, agent_counts, iso_type)
        ("2x1-2x1.txt", 4, [2], "isomorphic"),
        ("2x2-2x2.txt", 8, [2, 4], "isomorphic"),
        ("3x3-3x3.txt", 18, [2, 3, 6], "isomorphic"),
        ("4x2-4x2.txt", 16, [2, 4, 8], "isomorphic"),
        ("3x2-2x1.txt", 10, [2, 5], "non-isomorphic"),
        ("3comp-2x1.txt", 6, [2, 3], "3-component"),
        ("3x3-3x3-3x3.txt", 27, [3, 9], "3-component-iso"),
    ]

    for grid_file, S, agent_counts, iso_type in configs:
        path = _grid("disconnected", grid_file)
        for n in agent_counts:
            h = _feasible_horizons(S, n)
            if h < 1:
                continue
            for sym in [False, True]:
                for alg in ["ibis", "qibis"]:
                    label = f"{grid_file[:-4]}_n{n}_{alg}_sym{sym}"
                    scenarios.append(Scenario(
                        suite="E3", name=label, grid_path=path,
                        num_agents=n, horizon=h, algorithm=alg,
                        symmetry=sym,
                        description=f"Disconnected ({iso_type}): {grid_file}, n={n}, sym={sym}",
                    ))
    return scenarios


# ---------------------------------------------------------------------------
# E4: CEGAR vs direct synthesis
# ---------------------------------------------------------------------------

def e4_cegar_vs_direct() -> list[Scenario]:
    """Compare CEGAR (orbit/discrete) against direct IBIS/Q-IBIS.

    CEGAR with orbit partition should be faster on symmetric grids where
    the orbit partition produces few blocks (coarse abstraction).
    On asymmetric grids, orbit = discrete, so no benefit.
    """
    scenarios: list[Scenario] = []
    configs = [
        # Symmetric grids: orbit partition beneficial
        ("symmetric", "3x3.txt", 9, [3]),
        ("symmetric", "4x4.txt", 16, [2, 4]),
        ("symmetric", "4x3.txt", 12, [2, 3]),
        ("symmetric", "5x5.txt", 25, [5]),
        ("symmetric", "6x4.txt", 24, [2, 3]),
        # Asymmetric grids: orbit = discrete, CEGAR overhead
        ("asymmetric", "5x5-13.txt", 12, [2, 3, 4]),
        ("asymmetric", "interesting.txt", 12, [2, 3, 4]),
        # Local symmetry grids
        ("local_symmetry", "L-4x4-2.txt", 14, [2, 7]),
        ("local_symmetry", "L-6x6-6.txt", 30, [2, 3, 5]),
        # Disconnected grids
        ("disconnected", "2x2-2x2.txt", 8, [2, 4]),
        ("disconnected", "3x3-3x3.txt", 18, [2, 3]),
    ]

    for category, grid_file, S, agent_counts in configs:
        path = _grid(category, grid_file)
        for n in agent_counts:
            h = _feasible_horizons(S, n)
            if h < 1:
                continue
            # Direct algorithms
            for alg in ["ibis", "qibis"]:
                label = f"{grid_file[:-4]}_n{n}_{alg}"
                scenarios.append(Scenario(
                    suite="E4", name=label, grid_path=path,
                    num_agents=n, horizon=h, algorithm=alg,
                    symmetry=True,
                    description=f"Direct {alg}: {grid_file}, n={n}",
                ))
            # CEGAR with orbit and discrete
            for partition in ["orbit", "discrete"]:
                label = f"{grid_file[:-4]}_n{n}_cegar-ibis_{partition}"
                scenarios.append(Scenario(
                    suite="E4", name=label, grid_path=path,
                    num_agents=n, horizon=h, algorithm="cegar-ibis",
                    symmetry=True, partition=partition,
                    description=f"CEGAR-IBIS ({partition}): {grid_file}, n={n}",
                ))
    return scenarios


# ---------------------------------------------------------------------------
# E5: Q-IBIS as CEGAR synthesiser
# ---------------------------------------------------------------------------

def e5_cegar_synthesiser() -> list[Scenario]:
    """Compare CEGAR+IBIS vs CEGAR+Q-IBIS on the same grids.

    Q-IBIS uses RL-guided proposals which may converge faster on larger grids
    where the action space is too large for unguided best-response.
    """
    scenarios: list[Scenario] = []
    configs = [
        ("symmetric", "4x4.txt", 16, [2, 4]),
        ("symmetric", "5x4.txt", 20, [2, 4]),
        ("symmetric", "6x4.txt", 24, [2, 3]),
        ("local_symmetry", "L-6x6-6.txt", 30, [2, 3]),
        ("disconnected", "3x3-3x3.txt", 18, [2, 3]),
    ]

    for category, grid_file, S, agent_counts in configs:
        path = _grid(category, grid_file)
        for n in agent_counts:
            h = _feasible_horizons(S, n)
            if h < 1:
                continue
            for synthesiser in ["cegar-ibis", "cegar-qibis"]:
                for partition in ["orbit", "discrete"]:
                    label = f"{grid_file[:-4]}_n{n}_{synthesiser}_{partition}"
                    scenarios.append(Scenario(
                        suite="E5", name=label, grid_path=path,
                        num_agents=n, horizon=h, algorithm=synthesiser,
                        symmetry=True, partition=partition,
                        description=f"{synthesiser} ({partition}): {grid_file}, n={n}",
                    ))
    return scenarios


# ---------------------------------------------------------------------------
# E7: Scalability
# ---------------------------------------------------------------------------

def e7_scalability() -> list[Scenario]:
    """Increasing grid sizes to find the scalability limits of each approach.

    Fixed agent count (n=2), increasing territory size.
    Only grids where S is divisible by n (so all sectors can be claimed).
    """
    scenarios: list[Scenario] = []
    # Even-sector grids for n=2: S must be divisible by 2
    grids = [
        ("3x2.txt", 6), ("4x2.txt", 8), ("4x3.txt", 12),
        ("4x4.txt", 16), ("5x4.txt", 20), ("6x4.txt", 24),
        ("6x6.txt", 36),
    ]
    n = 2
    for grid_file, S in grids:
        path = _grid("symmetric", grid_file)
        h = _feasible_horizons(S, n)
        for alg in ["ibis", "qibis", "cegar-ibis", "cegar-qibis"]:
            for sym in [False, True]:
                label = f"{grid_file[:-4]}_n{n}_{alg}_sym{sym}"
                scenarios.append(Scenario(
                    suite="E7", name=label, grid_path=path,
                    num_agents=n, horizon=h, algorithm=alg,
                    symmetry=sym,
                    description=f"Scalability: {grid_file}, n={n}, {alg}, sym={sym}",
                ))
    return scenarios


# ---------------------------------------------------------------------------
# All scenarios
# ---------------------------------------------------------------------------

def all_scenarios(
    suites: list[str] | None = None,
) -> list[Scenario]:
    """Return all scenarios, optionally filtered by suite name."""
    generators = {
        "E1": e1_global_symmetry,
        "E2": e2_local_symmetry,
        "E3": e3_disconnected,
        "E4": e4_cegar_vs_direct,
        "E5": e5_cegar_synthesiser,
        "E7": e7_scalability,
        # E6 (random) is generated dynamically by gridgen
    }
    result: list[Scenario] = []
    for suite_name, gen in generators.items():
        if suites is not None and suite_name not in suites:
            continue
        result.extend(gen())
    return result
