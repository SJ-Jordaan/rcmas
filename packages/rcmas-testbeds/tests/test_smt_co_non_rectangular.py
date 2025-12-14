from __future__ import annotations

import io

from rcmas_core.engine import Territory

from rcmas_testbeds.testing.smt_co_sanity import assert_owner_history_monotone, assert_smt_co_debug_sanity
from rcmas_testbeds.testbeds.smt_co import solve_collective_optimality


def test_smt_co_non_rectangular_territory_is_sat_and_regions_make_sense() -> None:
    # Territory.from_ascii treats '.' as a sector; everything else is "not a sector".
    # This shape is intentionally non-rectangular / has holes.
    #
    # Coordinates (x,y) with '.' present:
    # y=0: (0,0) (1,0) (3,0)
    # y=1: (0,1) (2,1) (3,1)
    # y=2: (1,2)
    # y=3: (0,3) (1,3) (2,3)
    grid = """..#.
.#..
#.#
...#
"""

    territory = Territory.from_ascii(io.StringIO(grid))
    assert len(territory) == 10

    # Keep horizon small so this stays fast, but large enough to claim most cells.
    sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=5, debug=True)

    assert_smt_co_debug_sanity(sol=sol, territory=territory, num_agents=2)
    assert_owner_history_monotone(sol)

    # Rendering-related invariant: territory can be non-rectangular; ensure we actually have gaps.
    sectors = territory.ordered_sectors()
    max_x = max(c.x for c in sectors)
    max_y = max(c.y for c in sectors)
    # At least one coordinate in bounding box should be missing.
    sector_set = set(sectors)
    assert any(
        (x, y) not in {(c.x, c.y) for c in sector_set}
        for y in range(max_y + 1)
        for x in range(max_x + 1)
    )

    # Sanity: territory is truly non-rectangular.
    assert len(sector_set) < (max_x + 1) * (max_y + 1)
