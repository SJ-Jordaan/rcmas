from __future__ import annotations

from itertools import product

from rcmas_core.engine import Coord, GameState, Territory
from rcmas_testbeds.testbeds.smt_co import solve_collective_optimality


def _neighbors4_indexed(sectors: tuple[Coord, ...]) -> dict[int, set[int]]:
    index_by_coord = {c: i for i, c in enumerate(sectors)}
    nb: dict[int, set[int]] = {i: set() for i in range(len(sectors))}
    for i, c in enumerate(sectors):
        for cand in (
            Coord(c.x + 1, c.y),
            Coord(c.x - 1, c.y),
            Coord(c.x, c.y + 1),
            Coord(c.x, c.y - 1),
        ):
            j = index_by_coord.get(cand)
            if j is not None:
                nb[i].add(j)
    return nb


def _scores_from_owner(*, territory: Territory, owner_by_index: tuple[int, ...], num_agents: int) -> tuple[int, ...]:
    # Use the engine's scoring as ground truth for the intended objective.
    state = GameState(territory=territory, num_agents=num_agents, owner_by_index=owner_by_index, round_index=0)
    return state.scores()


def _apply_one_round(*, owner: tuple[int, ...], actions: tuple[int, ...]) -> tuple[int, ...]:
    """Apply one round with the same semantics as `smt_co.py`.

    - actions[a] is either -1 (no-op) or an index s into sectors
    - must not target owned sectors
    - must not collide (solver enforces distinctness)
    - if action[a] == s then owner[s] becomes a
    """

    next_owner = list(owner)
    for a, s in enumerate(actions):
        if s < 0:
            continue
        if next_owner[s] != -1:
            raise ValueError("invalid action targets owned sector")
        next_owner[s] = a
    return tuple(next_owner)


def test_smt_co_matches_bruteforce_oracle_tiny_instance() -> None:
    # 2x2 grid => 4 sectors; 2 agents; horizon 2 is small enough to brute force exhaustively.
    territory = Territory.from_ascii(["..", ".."])
    sectors = territory.ordered_sectors()

    num_agents = 2
    horizon = 2

    # Run SMT solver
    sol = solve_collective_optimality(territory=territory, num_agents=num_agents, horizon=horizon)
    assert sol.is_sat
    assert sol.actions_by_round is not None
    assert sol.final_state is not None

    # Convert solver actions to sector indices for comparison
    idx = {c: i for i, c in enumerate(sectors)}
    smt_plan: tuple[tuple[int, ...], ...] = tuple(
        tuple(-1 if act is None else idx[act] for act in step) for step in sol.actions_by_round
    )
    assert len(smt_plan) == horizon
    for step in smt_plan:
        assert len(step) == num_agents
        chosen = [s for s in step if s >= 0]
        assert len(chosen) == len(set(chosen))  # no collisions

    # Verify the plan respects the dynamics and leads to the returned final ownership.
    owner = tuple([-1] * len(sectors))
    for t in range(horizon):
        step = smt_plan[t]
        # Must not target owned sectors
        for a, s in enumerate(step):
            if s >= 0:
                assert owner[s] == -1
                assert 0 <= a < num_agents
        owner = _apply_one_round(owner=owner, actions=step)
    assert owner == sol.final_state.owner_by_index

    # Oracle: brute force all valid plans and compute the best total payoff.
    action_domain = tuple([-1] + list(range(len(sectors))))
    best_total = -1
    best_scores: set[tuple[int, ...]] = set()

    # Enumerate joint action profiles for each round.
    for round0 in product(action_domain, repeat=num_agents):
        if len([s for s in round0 if s >= 0]) != len({s for s in round0 if s >= 0}):
            continue
        owner0 = tuple([-1] * len(sectors))
        if any(s >= 0 and owner0[s] != -1 for s in round0):
            continue
        owner1 = _apply_one_round(owner=owner0, actions=round0)

        for round1 in product(action_domain, repeat=num_agents):
            if len([s for s in round1 if s >= 0]) != len({s for s in round1 if s >= 0}):
                continue
            # Cannot target already-owned sector
            if any(s >= 0 and owner1[s] != -1 for s in round1):
                continue
            owner2 = _apply_one_round(owner=owner1, actions=round1)
            scores = _scores_from_owner(territory=territory, owner_by_index=owner2, num_agents=num_agents)
            total = sum(scores)
            if total > best_total:
                best_total = total
                best_scores = {scores}
            elif total == best_total:
                best_scores.add(scores)

    assert best_total >= 0

    # Solver must achieve oracle-optimal total payoff.
    solver_scores = sol.final_state.scores()
    assert sum(solver_scores) == best_total
    assert solver_scores in best_scores


def test_smt_co_full_game_4x4_h8_reaches_max_score() -> None:
    # Full game for 4x4 with 2 agents: minimum horizon to fill is ceil(16/2)=8.
    # Exhaustive plan enumeration at h=8 is infeasible, so we assert a strong
    # necessary condition for correctness: the solver reaches a full board and
    # achieves the theoretical maximum total score (8+8=16) on an empty 4x4.
    territory = Territory.from_ascii(
        [
            "....",
            "....",
            "....",
            "....",
        ]
    )
    sectors = territory.ordered_sectors()
    num_agents = 2
    horizon = 8

    sol = solve_collective_optimality(territory=territory, num_agents=num_agents, horizon=horizon)
    assert sol.is_sat
    assert sol.actions_by_round is not None
    assert sol.final_state is not None

    idx = {c: i for i, c in enumerate(sectors)}
    smt_plan: tuple[tuple[int, ...], ...] = tuple(
        tuple(-1 if act is None else idx[act] for act in step) for step in sol.actions_by_round
    )

    # Validate plan obeys dynamics.
    owner = tuple([-1] * len(sectors))
    for t in range(horizon):
        step = smt_plan[t]
        chosen = [s for s in step if s >= 0]
        assert len(chosen) == len(set(chosen))
        for s in chosen:
            assert owner[s] == -1
        owner = _apply_one_round(owner=owner, actions=step)
    assert owner == sol.final_state.owner_by_index

    # Strong correctness checks for the full game.
    assert sol.final_state.is_terminal()
    solver_scores = sol.final_state.scores()
    assert solver_scores == (8, 8)
    assert sum(solver_scores) == 16
