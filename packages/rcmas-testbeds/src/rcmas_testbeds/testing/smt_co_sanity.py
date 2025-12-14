from __future__ import annotations

from collections import deque

from rcmas_core.engine import Coord, Territory

from rcmas_testbeds.testbeds.smt_co import SmtSolution


def is_connected_4n(points: set[Coord]) -> bool:
    if not points:
        return True

    start = next(iter(points))
    q: deque[Coord] = deque([start])
    seen = {start}

    while q:
        c = q.popleft()
        for nb in (Coord(c.x + 1, c.y), Coord(c.x - 1, c.y), Coord(c.x, c.y + 1), Coord(c.x, c.y - 1)):
            if nb in points and nb not in seen:
                seen.add(nb)
                q.append(nb)

    return seen == points


def assert_owner_history_monotone(sol: SmtSolution) -> None:
    assert sol.owner_by_round is not None, "debug=True required (owner_by_round missing)"

    for t in range(1, len(sol.owner_by_round)):
        prev = sol.owner_by_round[t - 1]
        cur = sol.owner_by_round[t]
        for s, (p, c) in enumerate(zip(prev, cur)):
            assert not (p != -1 and c == -1), f"sector {s} became unowned at t={t}"


def assert_smt_co_debug_sanity(
    *,
    sol: SmtSolution,
    territory: Territory,
    num_agents: int,
    expected_payoff_by_agent: tuple[int, ...] | None = None,
    require_terminal: bool = False,
) -> None:
    """Generic checks that the SMT debug variables are internally consistent.

    This is meant to be reusable across many territories/horizons:
    - if payoff[a] > 0, the witness region exists, has size==payoff[a], is connected,
      and all cells in it are owned by agent a in the final state.
    - payoff[a] equals max(size[*][a]) and matches the chosen best_seed.
    """

    assert sol.is_sat
    assert sol.final_state is not None

    if require_terminal:
        assert sol.final_state.is_terminal()

    assert sol.payoff_by_agent is not None, "debug=True required (payoff_by_agent missing)"
    assert sol.owner_by_round is not None, "debug=True required (owner_by_round missing)"
    assert sol.size_by_seed is not None, "debug=True required (size_by_seed missing)"
    assert sol.best_seed_by_agent is not None, "debug=True required (best_seed_by_agent missing)"
    assert sol.best_region_by_agent is not None, "debug=True required (best_region_by_agent missing)"

    if expected_payoff_by_agent is not None:
        assert sol.payoff_by_agent == expected_payoff_by_agent

    sectors = territory.ordered_sectors()
    idx = {c: i for i, c in enumerate(sectors)}
    sector_set = set(sectors)

    assert len(sol.payoff_by_agent) == num_agents
    assert len(sol.best_seed_by_agent) == num_agents
    assert len(sol.best_region_by_agent) == num_agents

    for a in range(num_agents):
        payoff = sol.payoff_by_agent[a]
        max_size = max(row[a] for row in sol.size_by_seed)
        assert payoff == max_size

        seed = sol.best_seed_by_agent[a]
        assert seed is not None

        if payoff <= 0:
            # The implementation intentionally returns no witness when size <= 0.
            assert sol.best_region_by_agent[a] is None
            continue

        assert sol.size_by_seed[seed][a] == payoff

        region = sol.best_region_by_agent[a]
        assert region is not None
        region_set = set(region)

        assert len(region_set) == payoff
        assert region_set.issubset(sector_set)

        for c in region_set:
            s = idx[c]
            assert sol.final_state.owner_by_index[s] == a

        assert is_connected_4n(region_set)
