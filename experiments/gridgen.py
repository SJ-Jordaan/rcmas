"""Random grid generator for E6 scenarios.

Generates grids with controlled properties:
- Size (bounding box and sector count)
- Connectivity (connected vs disconnected)
- Shape (rectangular, holes, L-shaped)
- Symmetry level (high Aut, low Aut, trivial Aut)

Each generated grid is written to a temp directory and paired with feasible
(agents, horizon) configurations to produce Scenario objects.
"""

from __future__ import annotations

import os
import random
from collections import deque

from .scenarios import Scenario


def _connected_component(grid: list[list[bool]], start: tuple[int, int]) -> set[tuple[int, int]]:
    """BFS to find a connected component in a 2D boolean grid."""
    rows, cols = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc]:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return visited


def _is_connected(grid: list[list[bool]]) -> bool:
    """Check if all True cells in the grid form a single connected component."""
    cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c]]
    if not cells:
        return True
    comp = _connected_component(grid, cells[0])
    return len(comp) == len(cells)


def generate_random_grid(
    width: int,
    height: int,
    density: float = 0.7,
    ensure_connected: bool = True,
    rng: random.Random | None = None,
) -> list[str]:
    """Generate a random ASCII grid.

    Args:
        width: Grid width (columns).
        height: Grid height (rows).
        density: Fraction of cells that are playable sectors (0.0 to 1.0).
        ensure_connected: If True, retry until the grid is connected.
        rng: Random number generator instance.

    Returns:
        List of ASCII strings suitable for Territory.from_ascii().
    """
    if rng is None:
        rng = random.Random()

    for _ in range(200):
        grid = [[rng.random() < density for _ in range(width)] for _ in range(height)]
        # Ensure at least 2 sectors
        sector_count = sum(sum(row) for row in grid)
        if sector_count < 2:
            continue
        if ensure_connected and not _is_connected(grid):
            continue
        lines = []
        for row in grid:
            lines.append("".join("." if cell else "#" for cell in row))
        return lines

    # Fallback: full rectangle
    return ["." * width] * height


def generate_disconnected_grid(
    comp_width: int,
    comp_height: int,
    num_components: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Generate a disconnected grid with N rectangular components.

    Components are placed side by side separated by wall columns.
    Optional random holes within each component.
    """
    if rng is None:
        rng = random.Random()

    parts: list[list[str]] = []
    for _ in range(num_components):
        comp = ["." * comp_width] * comp_height
        parts.append(comp)

    # Join with wall columns
    lines = []
    for row_idx in range(comp_height):
        row_parts = [parts[c][row_idx] for c in range(num_components)]
        lines.append("#".join(row_parts))
    return lines


def _grid_to_file(lines: list[str], directory: str, name: str) -> str:
    """Write grid lines to a file and return the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.txt")
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def _count_sectors(lines: list[str]) -> int:
    return sum(c == "." for line in lines for c in line)


def e6_random_scenarios(
    num_grids: int = 20,
    seed: int = 42,
    output_dir: str | None = None,
) -> list[Scenario]:
    """Generate E6 random scenarios.

    Creates a mix of:
    - Small connected grids (3x3 to 5x5, density 0.5-1.0)
    - Medium connected grids (5x5 to 7x7, density 0.5-0.9)
    - Disconnected grids (2-3 components)
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "grids", "random"
        )

    rng = random.Random(seed)
    scenarios: list[Scenario] = []

    grid_idx = 0

    # Small connected grids
    for _ in range(num_grids // 3):
        w = rng.randint(3, 5)
        h = rng.randint(3, 5)
        density = rng.uniform(0.5, 1.0)
        lines = generate_random_grid(w, h, density, ensure_connected=True, rng=rng)
        S = _count_sectors(lines)
        if S < 4:
            continue
        name = f"rand_c_{grid_idx:03d}_{w}x{h}_S{S}"
        path = _grid_to_file(lines, output_dir, name)
        grid_idx += 1

        # Generate feasible agent counts
        for n in range(2, min(S, 8)):
            if S % n != 0:
                continue
            horizon = S // n
            if horizon < 2:
                continue
            for sym in [False, True]:
                scenarios.append(Scenario(
                    suite="E6", name=f"{name}_n{n}_sym{sym}",
                    grid_path=path, num_agents=n, horizon=horizon,
                    algorithm="sibis", symmetry=sym,
                    description=f"Random connected: {name}, n={n}",
                ))

    # Medium connected grids
    for _ in range(num_grids // 3):
        w = rng.randint(5, 7)
        h = rng.randint(5, 7)
        density = rng.uniform(0.5, 0.9)
        lines = generate_random_grid(w, h, density, ensure_connected=True, rng=rng)
        S = _count_sectors(lines)
        if S < 6:
            continue
        name = f"rand_m_{grid_idx:03d}_{w}x{h}_S{S}"
        path = _grid_to_file(lines, output_dir, name)
        grid_idx += 1

        for n in [2, 3]:
            if S % n != 0:
                continue
            horizon = S // n
            if horizon < 2:
                continue
            for sym in [False, True]:
                scenarios.append(Scenario(
                    suite="E6", name=f"{name}_n{n}_sym{sym}",
                    grid_path=path, num_agents=n, horizon=horizon,
                    algorithm="sibis", symmetry=sym,
                    description=f"Random medium: {name}, n={n}",
                ))

    # Disconnected grids
    for _ in range(num_grids // 3):
        cw = rng.randint(2, 4)
        ch = rng.randint(2, 4)
        nc = rng.randint(2, 3)
        lines = generate_disconnected_grid(cw, ch, nc, rng=rng)
        S = _count_sectors(lines)
        if S < 4:
            continue
        name = f"rand_d_{grid_idx:03d}_{nc}x({cw}x{ch})_S{S}"
        path = _grid_to_file(lines, output_dir, name)
        grid_idx += 1

        for n in range(2, min(S, 8)):
            if S % n != 0:
                continue
            horizon = S // n
            if horizon < 2:
                continue
            for sym in [False, True]:
                scenarios.append(Scenario(
                    suite="E6", name=f"{name}_n{n}_sym{sym}",
                    grid_path=path, num_agents=n, horizon=horizon,
                    algorithm="sibis", symmetry=sym,
                    description=f"Random disconnected: {name}, n={n}",
                ))

    return scenarios
