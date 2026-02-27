"""Integration tests for smt_solve.py.

Includes oracle tests, smoke tests, sanity checks,
and the fixed-policy correctness test.
"""

from __future__ import annotations

import io
from collections import deque
from itertools import product

import pytest

from rcmas.model import Coord, State, Territory, scores
from rcmas.smt_solve import SmtSolution, solve_collective_optimality, solve_smt_game

z3 = pytest.importorskip("z3")


# ---------------------------------------------------------------------------
# Sanity-check helpers
# ---------------------------------------------------------------------------

def _is_connected_4n(points: set[Coord]) -> bool:
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


def _assert_owner_history_monotone(sol: SmtSolution) -> None:
    assert sol.owner_by_round is not None
    for t in range(1, len(sol.owner_by_round)):
        prev = sol.owner_by_round[t - 1]
        cur = sol.owner_by_round[t]
        for s, (p, c) in enumerate(zip(prev, cur)):
            assert not (p != -1 and c == -1), f"sector {s} became unowned at t={t}"


def _assert_debug_sanity(
    *,
    sol: SmtSolution,
    territory: Territory,
    num_agents: int,
    expected_payoff: tuple[int, ...] | None = None,
    require_terminal: bool = False,
) -> None:
    assert sol.is_sat
    assert sol.final_state is not None
    if require_terminal:
        assert sol.final_state.is_terminal()
    assert sol.payoff_by_agent is not None
    assert sol.owner_by_round is not None
    assert sol.size_by_seed is not None
    assert sol.best_seed_by_agent is not None
    assert sol.best_region_by_agent is not None

    if expected_payoff is not None:
        assert sol.payoff_by_agent == expected_payoff

    sectors = territory.ordered_sectors()
    idx = {c: i for i, c in enumerate(sectors)}
    sector_set = set(sectors)

    assert len(sol.payoff_by_agent) == num_agents
    for a in range(num_agents):
        payoff = sol.payoff_by_agent[a]
        max_size = max(row[a] for row in sol.size_by_seed)
        assert payoff == max_size

        seed = sol.best_seed_by_agent[a]
        assert seed is not None

        if payoff <= 0:
            assert sol.best_region_by_agent[a] is None
            continue

        assert sol.size_by_seed[seed][a] == payoff
        region = sol.best_region_by_agent[a]
        assert region is not None
        region_set = set(region)
        assert len(region_set) == payoff
        assert region_set.issubset(sector_set)
        for c in region_set:
            assert sol.final_state.owner_by_index[idx[c]] == a
        assert _is_connected_4n(region_set)


# ===================================================================
# Smoke tests
# ===================================================================

class TestSmokeTests:
    def test_reaches_terminal_on_tiny_grid(self):
        territory = Territory.from_ascii(["..", ".."])
        sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=2)
        assert sol.is_sat
        assert sol.final_state is not None
        assert sol.final_state.is_terminal()
        assert len(scores(sol.final_state)) == 2

    def test_single_agent_single_sector(self):
        """1 agent, 1 sector, 1 round: trivially fills the board."""
        territory = Territory.from_ascii(["."])
        sol = solve_collective_optimality(territory=territory, num_agents=1, horizon=1)
        assert sol.is_sat
        assert sol.final_state is not None
        assert sol.final_state.is_terminal()

    def test_single_agent_linear_territory(self):
        """1 agent on a 1x3 territory, 3 rounds: claims all sectors."""
        territory = Territory.from_ascii(["..."])
        sol = solve_collective_optimality(territory=territory, num_agents=1, horizon=3)
        assert sol.is_sat
        assert sol.final_state is not None
        assert sol.final_state.is_terminal()

    def test_two_agents_fill_2x2(self):
        territory = Territory.from_ascii(["..", ".."])
        sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=2)
        assert sol.is_sat
        assert sol.final_state is not None
        assert sol.final_state.is_terminal()

    def test_unsat_when_too_few_sectors(self):
        """2 agents on 1 sector, 1 round: both must claim but only 1 sector -> unsat."""
        territory = Territory.from_ascii(["."])
        sol = solve_smt_game(
            territory=territory, num_agents=2, horizon=1,
            objective="sum", require_victory=False,
        )
        assert not sol.is_sat


# ===================================================================
# Debug sanity tests
# ===================================================================

class TestDebugSanity:
    def test_4x4_h8(self):
        territory = Territory.from_ascii(["....", "....", "....", "...."])
        sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=8, debug=True)
        _assert_debug_sanity(
            sol=sol, territory=territory, num_agents=2,
            expected_payoff=(8, 8), require_terminal=True,
        )
        _assert_owner_history_monotone(sol)

    @pytest.mark.parametrize(
        ("grid", "agents", "horizon", "expected_payoff", "require_terminal"),
        [
            ("....\n....\n....\n....\n", 2, 8, (8, 8), True),
            ("..#.\n.#..\n#.#\n...#\n", 2, 5, None, False),
        ],
    )
    def test_parametrized(self, grid, agents, horizon, expected_payoff, require_terminal):
        territory = Territory.from_ascii(io.StringIO(grid))
        sol = solve_collective_optimality(territory=territory, num_agents=agents, horizon=horizon, debug=True)
        _assert_debug_sanity(
            sol=sol, territory=territory, num_agents=agents,
            expected_payoff=expected_payoff, require_terminal=require_terminal,
        )
        _assert_owner_history_monotone(sol)

    def test_non_rectangular(self):
        grid = "..#.\n.#..\n#.#\n...#\n"
        territory = Territory.from_ascii(io.StringIO(grid))
        assert len(territory) == 10
        sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=5, debug=True)
        _assert_debug_sanity(sol=sol, territory=territory, num_agents=2)
        _assert_owner_history_monotone(sol)

    def test_owner_history_length(self):
        """Owner history has T+1 snapshots (initial + one per round)."""
        territory = Territory.from_ascii(["..", ".."])
        sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=2, debug=True)
        assert sol.owner_by_round is not None
        assert len(sol.owner_by_round) == 3  # t=0, t=1, t=2

    def test_actions_by_round_shape(self):
        """actions_by_round has T entries, each with A actions."""
        territory = Territory.from_ascii(["..", ".."])
        sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=2, debug=True)
        assert sol.actions_by_round is not None
        assert len(sol.actions_by_round) == 2
        for step in sol.actions_by_round:
            assert len(step) == 2


# ===================================================================
# Fixed-policy correctness tests
# ===================================================================

class TestFixedPolicy:
    def test_fixed_agent_follows_policy(self):
        """When agent 1 is fixed to claim (0,0), it must claim (0,0)."""
        territory = Territory.from_ascii(["..", ".."])
        initial_state = (-1, -1, -1, -1)
        fixed_policy = {initial_state: Coord(0, 0)}

        sol = solve_smt_game(
            territory=territory, num_agents=2, horizon=2,
            objective=0,
            fixed_policy_by_agent=(None, fixed_policy),
            require_victory=True, debug=True,
        )
        assert sol.is_sat
        assert sol.actions_by_round is not None
        assert sol.actions_by_round[0][1] == Coord(0, 0)

    def test_fixed_suboptimal_policy_lowers_payoff(self):
        """Fix agent 1 to a sub-optimal (disconnected) strategy on 2x2.

        Sectors: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
        Agent 1 is forced to claim diagonals (0,0) then (1,1) -> disconnected -> payoff 1.
        Agent 0 is free and gets the remaining connected pair -> payoff 2.
        """
        territory = Territory.from_ascii(["..", ".."])
        initial_state = (-1, -1, -1, -1)
        # After round 0: agent 0 claims something, agent 1 claims (0,0)
        # We fix agent 1's policy for both rounds
        fixed_policy = {
            initial_state: Coord(0, 0),
        }

        fixed_sol = solve_smt_game(
            territory=territory, num_agents=2, horizon=2,
            objective=0,
            fixed_policy_by_agent=(None, fixed_policy),
            require_victory=True, debug=True,
        )
        assert fixed_sol.is_sat
        assert fixed_sol.payoff_by_agent is not None
        # Agent 0 is optimized, so it should get at least payoff 2
        assert fixed_sol.payoff_by_agent[0] >= 2

    def test_free_vs_fixed_comparison(self):
        """On 2x1 (2 sectors), 2 agents, 1 round: both claim a sector.
        Fixed: agent 1 claims sector 1. Free agent 0 gets sector 0.
        Free: solver picks the optimal allocation."""
        territory = Territory.from_ascii([".."])
        initial_state = (-1, -1)

        # Fixed: agent 1 must claim sector 1 = Coord(1, 0)
        fixed_sol = solve_smt_game(
            territory=territory, num_agents=2, horizon=1,
            objective=0,
            fixed_policy_by_agent=(None, {initial_state: Coord(1, 0)}),
            require_victory=True, debug=True,
        )
        assert fixed_sol.is_sat
        assert fixed_sol.actions_by_round is not None
        assert fixed_sol.actions_by_round[0][1] == Coord(1, 0)
        assert fixed_sol.actions_by_round[0][0] == Coord(0, 0)

        # Free: no constraints
        free_sol = solve_smt_game(
            territory=territory, num_agents=2, horizon=1,
            objective=0, require_victory=True, debug=True,
        )
        assert free_sol.is_sat
        assert free_sol.payoff_by_agent is not None
        assert free_sol.payoff_by_agent[0] == 1


# ===================================================================
# Input validation
# ===================================================================

class TestInputValidation:
    def test_zero_agents_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="num_agents"):
            solve_smt_game(territory=territory, num_agents=0, horizon=1, objective="sum")

    def test_zero_horizon_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="horizon"):
            solve_smt_game(territory=territory, num_agents=1, horizon=0, objective="sum")

    def test_invalid_objective_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="objective"):
            solve_smt_game(territory=territory, num_agents=1, horizon=1, objective="bad")

    def test_negative_timeout_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="timeout"):
            solve_smt_game(territory=territory, num_agents=1, horizon=1, objective="sum", timeout_ms=-1)


# ===================================================================
# Oracle: brute-force validation
# ===================================================================

def _apply_one_round(*, owner: tuple[int, ...], actions: tuple[int, ...]) -> tuple[int, ...]:
    next_owner = list(owner)
    for a, s in enumerate(actions):
        if s < 0:
            continue
        if next_owner[s] != -1:
            raise ValueError("invalid action targets owned sector")
        next_owner[s] = a
    return tuple(next_owner)


class TestBruteForceOracle:
    def test_2x2_matches_oracle(self):
        """Exhaustive brute-force on 2x2, 2 agents, 2 rounds.

        Since action >= 0 (no no-op), every agent must claim a sector each round.
        On 2x2 with 2 agents and 2 rounds: 4 claims = 4 sectors, exactly fills.
        """
        territory = Territory.from_ascii(["..", ".."])
        sectors = territory.ordered_sectors()
        num_agents = 2
        horizon = 2
        S = len(sectors)

        sol = solve_collective_optimality(territory=territory, num_agents=num_agents, horizon=horizon)
        assert sol.is_sat
        assert sol.actions_by_round is not None
        assert sol.final_state is not None

        idx = {c: i for i, c in enumerate(sectors)}
        smt_plan = tuple(
            tuple(idx[act] for act in step) for step in sol.actions_by_round
        )

        # Validate plan dynamics
        owner = tuple([-1] * S)
        for t in range(horizon):
            step = smt_plan[t]
            chosen = list(step)
            assert len(chosen) == len(set(chosen)), "collision in SMT solution"
            for s in chosen:
                assert owner[s] == -1, "reclaim in SMT solution"
            owner = _apply_one_round(owner=owner, actions=step)
        assert owner == sol.final_state.owner_by_index

        # Brute force: enumerate all valid action sequences (no no-op)
        action_domain = list(range(S))
        best_total = -1
        best_scores: set[tuple[int, ...]] = set()

        for round0 in product(action_domain, repeat=num_agents):
            if len(set(round0)) != num_agents:
                continue
            owner0 = tuple([-1] * S)
            if any(owner0[s] != -1 for s in round0):
                continue
            owner1 = _apply_one_round(owner=owner0, actions=round0)

            for round1 in product(action_domain, repeat=num_agents):
                if len(set(round1)) != num_agents:
                    continue
                if any(owner1[s] != -1 for s in round1):
                    continue
                owner2 = _apply_one_round(owner=owner1, actions=round1)
                sc = scores(State(territory=territory, num_agents=num_agents, owner_by_index=owner2, round_index=0))
                total = sum(sc)
                if total > best_total:
                    best_total = total
                    best_scores = {sc}
                elif total == best_total:
                    best_scores.add(sc)

        solver_scores = scores(sol.final_state)
        assert sum(solver_scores) == best_total
        assert solver_scores in best_scores

    def test_1x4_single_agent_matches_oracle(self):
        """1 agent on 1x4, 4 rounds: claims all 4 sectors."""
        territory = Territory.from_ascii(["...."])
        sectors = territory.ordered_sectors()
        num_agents = 1
        horizon = 4
        S = len(sectors)

        sol = solve_collective_optimality(territory=territory, num_agents=num_agents, horizon=horizon)
        assert sol.is_sat
        assert sol.final_state is not None
        assert sol.final_state.is_terminal()
        # Single agent owns all 4 connected sectors -> payoff 4
        solver_scores = scores(sol.final_state)
        assert solver_scores == (4,)

    def test_4x4_h8_max_score(self):
        territory = Territory.from_ascii(["....", "....", "....", "...."])
        num_agents = 2
        horizon = 8

        sol = solve_collective_optimality(territory=territory, num_agents=num_agents, horizon=horizon)
        assert sol.is_sat
        assert sol.final_state is not None
        assert sol.final_state.is_terminal()
        solver_scores = scores(sol.final_state)
        assert solver_scores == (8, 8)
        assert sum(solver_scores) == 16
