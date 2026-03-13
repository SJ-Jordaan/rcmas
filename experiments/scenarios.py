"""Experiment scenario definitions for symmetry reduction evaluation.

Each scenario is a (grid, agents, horizon, algorithm, flags) tuple with metadata
explaining *why* it's included — which feature it tests, what the control is, and
what we expect to see.

All suites use S-IBIS exclusively (SAT-based best-response).

Scenario suites:
    E1 — Global symmetry breaking (rectangular grids, D4/D2 Aut)
    E2 — Local symmetry breaking (L-shapes, trivial global Aut)
    E3 — Disconnected territory handling (multi-component grids)
    E4 — Asymmetric control (grids with trivial Aut, overhead measurement)
    E5 — Scalability (increasing grid sizes, n=2/3/4 tracks)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

GRIDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grids")


@dataclass(frozen=True, slots=True)
class Scenario:
    """A single benchmark scenario."""

    suite: str
    name: str
    grid_path: str
    num_agents: int
    horizon: int
    algorithm: str  # "sibis"
    symmetry: bool
    partition: str = "orbit"  # only for cegar algorithms (unused here)
    demands: tuple[int, ...] | None = None
    description: str = ""


def _grid(category: str, name: str) -> str:
    return os.path.join(GRIDS_DIR, category, name)


def _feasible_horizons(S: int, n: int) -> int:
    """Horizon = S / n.  Requires S divisible by n (for require_victory)."""
    if S % n != 0:
        raise ValueError(f"S={S} not divisible by n={n}")
    return S // n


def _sibis(
    suite: str, grid_file: str, path: str, n: int, h: int, sym: bool, desc: str,
) -> Scenario:
    label = f"{grid_file[:-4]}_n{n}_sym{sym}"
    return Scenario(
        suite=suite, name=label, grid_path=path,
        num_agents=n, horizon=h, algorithm="sibis",
        symmetry=sym, description=desc,
    )


# ---------------------------------------------------------------------------
# E1: Global symmetry breaking
# ---------------------------------------------------------------------------

def e1_global_symmetry() -> list[Scenario]:
    """Rectangular grids with D4/D2 automorphisms. Compare sym on/off.

    These grids have non-trivial Aut(T):
    - Squares (NxN): |Aut| = 8 (D4)
    - Rectangles (NxM, N!=M): |Aut| = 4 (V4 ≅ Klein four-group)

    Symmetry breaking should prune equivalent search branches.
    """
    scenarios: list[Scenario] = []
    configs = [
        # (grid_file, S, agent_counts)
        ("3x3.txt", 9, [3]),
        ("4x3.txt", 12, [2, 3, 4]),
        ("4x4.txt", 16, [2, 4]),
        ("5x3.txt", 15, [3, 5]),
        ("5x4.txt", 20, [2, 4, 5]),
        ("5x5.txt", 25, [5]),
        ("6x3.txt", 18, [2, 3, 6]),
        ("6x4.txt", 24, [2, 3, 4]),
        ("6x5.txt", 30, [2, 3, 5, 6]),
        ("6x6.txt", 36, [2, 3, 4, 6]),
    ]

    for grid_file, S, agent_counts in configs:
        path = _grid("symmetric", grid_file)
        for n in agent_counts:
            h = _feasible_horizons(S, n)
            for sym in [False, True]:
                scenarios.append(_sibis(
                    "E1", grid_file, path, n, h, sym,
                    f"Global symmetry: {grid_file} with {n} agents, sym={sym}",
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
        ("L-6x4-4.txt", 16, [2, 4, 8]),
        ("L-6x6-6.txt", 30, [2, 3, 5]),
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
                scenarios.append(_sibis(
                    "E2", grid_file, path, n, h, sym,
                    f"Local symmetry: {grid_file} with {n} agents, sym={sym}",
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
                scenarios.append(_sibis(
                    "E3", grid_file, path, n, h, sym,
                    f"Disconnected ({iso_type}): {grid_file}, n={n}, sym={sym}",
                ))
    return scenarios


# ---------------------------------------------------------------------------
# E4: Asymmetric control (overhead measurement)
# ---------------------------------------------------------------------------

def e4_asymmetric_control() -> list[Scenario]:
    """Asymmetric grids with |Aut(T)| = 1. Compare sym on/off.

    When the territory has no non-trivial automorphisms, the spatial
    canonicalization constraint is vacuous (all sectors are their own orbit
    representative).  Agent lex-leader adds at most n-1 constraints.

    Purpose: demonstrate that symmetry breaking introduces negligible
    overhead when there is no symmetry to exploit.
    """
    scenarios: list[Scenario] = []
    configs = [
        # (grid_file, S, agent_counts)
        ("interesting.txt", 12, [2, 3, 4]),
        ("5x5-13.txt", 12, [2, 3, 4]),
        ("6x6-18.txt", 16, [2, 4]),
        ("5x5-shaped.txt", 24, [2, 3, 4]),
        ("8x8-34.txt", 24, [2, 3, 4]),
    ]

    for grid_file, S, agent_counts in configs:
        path = _grid("asymmetric", grid_file)
        for n in agent_counts:
            h = _feasible_horizons(S, n)
            for sym in [False, True]:
                scenarios.append(_sibis(
                    "E4", grid_file, path, n, h, sym,
                    f"Asymmetric control: {grid_file}, n={n}, sym={sym}",
                ))
    return scenarios


# ---------------------------------------------------------------------------
# E5: Scalability
# ---------------------------------------------------------------------------

def e5_scalability() -> list[Scenario]:
    """Increasing grid sizes to find the scalability limits.

    Three agent-count tracks on symmetric rectangular grids:
    - n=2: increasing even-S grids (S = 6..36)
    - n=3: increasing S%%3==0 grids (S = 9..36)
    - n=4: increasing S%%4==0 grids (S = 8..36)

    Each track produces a clean scaling curve: fixed n, growing S.
    """
    scenarios: list[Scenario] = []
    tracks = [
        # (n, [(grid_file, S), ...])
        (2, [
            ("3x2.txt", 6), ("4x2.txt", 8), ("5x2.txt", 10),
            ("4x3.txt", 12), ("7x2.txt", 14), ("4x4.txt", 16),
            ("5x4.txt", 20), ("6x4.txt", 24), ("7x4.txt", 28),
            ("8x4.txt", 32), ("6x6.txt", 36),
        ]),
        (3, [
            ("3x3.txt", 9), ("4x3.txt", 12), ("5x3.txt", 15),
            ("6x3.txt", 18), ("7x3.txt", 21), ("6x4.txt", 24),
            ("9x3.txt", 27), ("6x5.txt", 30), ("6x6.txt", 36),
        ]),
        (4, [
            ("4x2.txt", 8), ("4x3.txt", 12), ("4x4.txt", 16),
            ("5x4.txt", 20), ("6x4.txt", 24), ("7x4.txt", 28),
            ("8x4.txt", 32), ("6x6.txt", 36),
        ]),
    ]

    for n, grids in tracks:
        for grid_file, S in grids:
            path = _grid("symmetric", grid_file)
            h = _feasible_horizons(S, n)
            for sym in [False, True]:
                scenarios.append(_sibis(
                    "E5", grid_file, path, n, h, sym,
                    f"Scalability: {grid_file}, n={n}, sym={sym}",
                ))
    return scenarios


# ---------------------------------------------------------------------------
# All scenarios
# ---------------------------------------------------------------------------

def all_scenarios(
    suites: list[str] | None = None,
) -> list[Scenario]:
    """Return all scenarios, optionally filtered by suite name."""
    from .gridgen import e6_random_scenarios

    generators = {
        "E1": e1_global_symmetry,
        "E2": e2_local_symmetry,
        "E3": e3_disconnected,
        "E4": e4_asymmetric_control,
        "E5": e5_scalability,
        "E6": e6_random_scenarios,
    }
    result: list[Scenario] = []
    for suite_name, gen in generators.items():
        if suites is not None and suite_name not in suites:
            continue
        result.extend(gen())
    return result
