"""Tests for smt_solve.py (thesis Sections 4.5-4.8).

Includes oracle tests of the best-response query ``smt(G_a(pi), reward_a)``
against exhaustive enumeration of each agent's deviations from the
canonical completion (Lemma 4.7.2), also where the number of agents does
not divide the number of sectors (3x3 with 2 agents, 5x2 with 3 agents).
"""

from __future__ import annotations

import random
import warnings
from collections import deque
from itertools import product

import pytest

import rcmas.smt_solve as smt_solve
from rcmas.model import Coord, State, Territory, rewards_from_owner, scores
from rcmas.smt_solve import (
    SmtSolution,
    best_response_binsearch_from_base,
    best_response_from_base,
    build_base_encoding,
    check_improvement_from_base,
    find_feasible_from_base,
    solve_collective_optimality,
    solve_from_base,
    solve_satisfying,
    solve_smt_game,
)
from rcmas.strategy import CanonicalCompletion, Play

from .brute_force import best_deviation, random_play

z3 = pytest.importorskip("z3")

RUNNING = ["..#.", "#...", ".#.#", "...#", "#.##"]


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
        for nb in (
            Coord(c.x + 1, c.y),
            Coord(c.x - 1, c.y),
            Coord(c.x, c.y + 1),
            Coord(c.x, c.y - 1),
        ):
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
            assert not (p != -1 and c != p), f"sector {s} changed controller at t={t}"


def _assert_debug_sanity(
    *, sol: SmtSolution, territory: Territory, num_agents: int
) -> None:
    """Rewards are those of the final state; witnesses are connected and owned."""
    assert sol.is_sat
    assert sol.final_state is not None
    assert sol.payoff_by_agent == scores(sol.final_state)
    assert sol.encoded_reward_by_agent is not None
    assert sol.best_region_by_agent is not None
    assert sol.best_seed_by_agent is not None
    idx = {c: i for i, c in enumerate(territory.ordered_sectors())}
    for a in range(num_agents):
        enc = sol.encoded_reward_by_agent[a]
        assert enc <= sol.payoff_by_agent[a]  # Lemma 4.3.1
        region = sol.best_region_by_agent[a]
        if enc == 0:
            assert region is None
            continue
        assert region is not None and len(region) == enc
        assert all(sol.final_state.owner_by_index[idx[c]] == a for c in region)
        assert _is_connected_4n(set(region))
        root = sol.best_seed_by_agent[a]
        assert root == min(idx[c] for c in region)


# ===================================================================
# Satisfying and collectively optimal synthesis (Theorems 4.5.1, 4.6.1)
# ===================================================================


class TestSmokeTests:
    def test_reaches_terminal_on_tiny_grid(self):
        territory = Territory.from_ascii(["..", ".."])
        sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=2)
        assert sol.is_sat
        assert sol.final_state is not None
        assert sol.final_state.is_terminal()
        assert sol.payoff_by_agent == (2, 2)

    def test_single_agent_single_sector(self):
        territory = Territory.from_ascii(["."])
        sol = solve_collective_optimality(territory=territory, num_agents=1, horizon=1)
        assert sol.is_sat
        assert sol.final_state is not None and sol.final_state.is_terminal()

    def test_single_agent_linear_territory(self):
        territory = Territory.from_ascii(["...."])
        sol = solve_collective_optimality(territory=territory, num_agents=1, horizon=4)
        assert sol.payoff_by_agent == (4,)

    def test_unsat_when_too_few_sectors(self):
        """2 agents, 1 sector, 1 round: both must claim, one sector only."""
        territory = Territory.from_ascii(["."])
        sol = solve_smt_game(
            territory=territory, num_agents=2, horizon=1, objective="sum"
        )
        assert not sol.is_sat
        assert sol.reason == "unsat"

    def test_running_example_optimum_is_11(self):
        """Section 3.10.2: the running example has a satisfying strategy with rewards summing to 11."""
        territory = Territory.from_ascii(RUNNING)
        for demands in (None, (2, 2, 2)):
            sol = solve_collective_optimality(
                territory=territory, num_agents=3, horizon=4, demands=demands
            )
            assert sol.payoff_by_agent is not None
            assert sum(sol.payoff_by_agent) == 11

    def test_satisfying_meets_demands(self):
        territory = Territory.from_ascii(["...", "...", "..."])
        sol = solve_satisfying(
            territory=territory, num_agents=3, horizon=3, demands=(3, 2, 1)
        )
        assert sol.is_sat and sol.payoff_by_agent is not None
        assert all(r >= d for r, d in zip(sol.payoff_by_agent, (3, 2, 1)))

    def test_unsatisfiable_demands(self):
        """No play of 3x3 with 3 agents gives every agent 4 (m = 3)."""
        territory = Territory.from_ascii(["...", "...", "..."])
        sol = solve_satisfying(
            territory=territory, num_agents=3, horizon=3, demands=(4, 0, 0)
        )
        assert not sol.is_sat and sol.reason == "unsat"

    def test_horizon_zero(self):
        """|T| < |Agt|: m = 0; satisfiable iff every demand is 0."""
        territory = Territory.from_ascii(["."])
        base = build_base_encoding(territory=territory, num_agents=2, horizon=0)
        sol = find_feasible_from_base(base)
        assert sol.is_sat and sol.payoff_by_agent == (0, 0)
        base = build_base_encoding(
            territory=territory, num_agents=2, horizon=0, demands=(1, 0)
        )
        assert find_feasible_from_base(base).reason == "unsat"


class TestDebugSanity:
    @pytest.mark.parametrize(
        "grid,agents",
        [
            (["....", "....", "....", "...."], 2),
            (["..#.", ".#..", "#.#", "...#"], 2),
            (RUNNING, 3),
        ],
    )
    def test_collective_optimum_witnesses(self, grid, agents):
        territory = Territory.from_ascii(grid)
        sol = solve_collective_optimality(
            territory=territory,
            num_agents=agents,
            horizon=len(territory) // agents,
            debug=True,
        )
        _assert_debug_sanity(sol=sol, territory=territory, num_agents=agents)
        # At the optimum of f every witness is exact (Corollary 4.3.3).
        assert sol.encoded_reward_by_agent == sol.payoff_by_agent
        _assert_owner_history_monotone(sol)

    def test_play_matches_trace(self):
        territory = Territory.from_ascii(["..", ".."])
        sol = solve_collective_optimality(territory=territory, num_agents=2, horizon=2)
        assert sol.play is not None and sol.owner_by_round is not None
        assert sol.play.states == sol.owner_by_round
        assert sol.play.to_coords(territory) == sol.actions_by_round
        assert Play.from_actions(4, 2, sol.play.actions) == sol.play


class TestFixedPolicy:
    def test_fixed_agent_follows_policy(self):
        territory = Territory.from_ascii(["..", ".."])
        fixed_policy = {(-1, -1, -1, -1): Coord(0, 0)}
        sol = solve_smt_game(
            territory=territory,
            num_agents=2,
            horizon=2,
            objective=0,
            fixed_policy_by_agent=(None, fixed_policy),
        )
        assert sol.is_sat and sol.actions_by_round is not None
        assert sol.actions_by_round[0][1] == Coord(0, 0)

    def test_free_vs_fixed_comparison(self):
        territory = Territory.from_ascii([".."])
        fixed_sol = solve_smt_game(
            territory=territory,
            num_agents=2,
            horizon=1,
            objective=0,
            fixed_policy_by_agent=(None, {(-1, -1): Coord(1, 0)}),
        )
        assert fixed_sol.actions_by_round == ((Coord(0, 0), Coord(1, 0)),)
        free_sol = solve_smt_game(
            territory=territory, num_agents=2, horizon=1, objective=0
        )
        assert free_sol.payoff_by_agent is not None and free_sol.payoff_by_agent[0] == 1


class TestInputValidation:
    def test_zero_agents_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="num_agents"):
            solve_smt_game(
                territory=territory, num_agents=0, horizon=1, objective="sum"
            )

    def test_zero_horizon_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="horizon"):
            solve_smt_game(
                territory=territory, num_agents=1, horizon=0, objective="sum"
            )

    def test_invalid_objective_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="objective"):
            solve_smt_game(
                territory=territory, num_agents=1, horizon=1, objective="bad"
            )

    def test_negative_timeout_raises(self):
        territory = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="timeout"):
            solve_smt_game(
                territory=territory,
                num_agents=1,
                horizon=1,
                objective="sum",
                timeout_ms=-1,
            )


def _apply_one_round(
    owner: tuple[int, ...], actions: tuple[int, ...]
) -> tuple[int, ...]:
    nxt = list(owner)
    for a, s in enumerate(actions):
        assert nxt[s] == -1
        nxt[s] = a
    return tuple(nxt)


class TestBruteForceOracle:
    def test_2x2_matches_oracle(self):
        """Exhaustive enumeration of the plays of 2x2 with 2 agents."""
        territory = Territory.from_ascii(["..", ".."])
        S, n = 4, 2
        sol = solve_collective_optimality(territory=territory, num_agents=n, horizon=2)
        assert sol.final_state is not None
        best = -1
        for r0 in product(range(S), repeat=n):
            if len(set(r0)) != n:
                continue
            o1 = _apply_one_round(tuple([-1] * S), r0)
            for r1 in product(range(S), repeat=n):
                if len(set(r1)) != n or any(o1[s] != -1 for s in r1):
                    continue
                o2 = _apply_one_round(o1, r1)
                sc = scores(State(territory=territory, num_agents=n, owner_by_index=o2))
                best = max(best, sum(sc))
        assert sum(scores(sol.final_state)) == best


# ===================================================================
# The cached formula F (Section 7.1)
# ===================================================================


class TestBaseEncoding:
    def test_symmetry_flag_has_no_effect_on_base(self):
        territory = Territory.from_ascii(["...", "...", "..."])
        plain = build_base_encoding(territory=territory, num_agents=3, horizon=3)
        with pytest.warns(DeprecationWarning):
            flagged = build_base_encoding(
                territory=territory, num_agents=3, horizon=3, symmetry_breaking=True
            )
        assert len(plain.constraints) == len(flagged.constraints)

    def test_demands_posted_in_base(self):
        territory = Territory.from_ascii(["...", "..."])
        base = build_base_encoding(
            territory=territory, num_agents=2, horizon=3, demands=(3, 3)
        )
        sol = find_feasible_from_base(base)
        assert sol.is_sat and sol.payoff_by_agent == (3, 3)

    def test_max_reward(self):
        territory = Territory.from_ascii(["...", "..."])
        assert (
            build_base_encoding(territory=territory, num_agents=2, horizon=3).max_reward
            == 3
        )
        abstract = Territory.from_ascii(["...."])
        base = build_base_encoding(
            territory=abstract,
            num_agents=2,
            horizon=2,
            weights=(1, 4, 2, 3),
            custom_neighbors={0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
        )
        assert base.max_reward == 7

    def test_weighted_rewards_reported_on_final_state(self):
        abstract = Territory.from_ascii(["..."])
        nb = {0: [2], 1: [], 2: [0]}  # block adjacency is not the synthetic row
        base = build_base_encoding(
            territory=abstract,
            num_agents=1,
            horizon=3,
            weights=(2, 3, 4),
            custom_neighbors=nb,
        )
        sol = solve_from_base(base, objective=0)
        assert sol.payoff_by_agent == (6,)
        assert sol.encoded_reward_by_agent == (6,)

    def test_symmetry_constraints_only_where_asked(self):
        """Lex-leader holds in the initial query with symmetry, and is absent without."""
        territory = Territory.from_ascii(["...", "...", "..."])
        base = build_base_encoding(territory=territory, num_agents=2, horizon=4)
        v = base.variables
        with_sym = find_feasible_from_base(
            base,
            symmetry=True,
            extra_constraints=lambda s, _v: s.add(v.action[0][0] > v.action[1][0]),
        )
        assert with_sym.reason == "unsat"
        without = find_feasible_from_base(
            base, extra_constraints=lambda s, _v: s.add(v.action[0][0] > v.action[1][0])
        )
        assert without.is_sat


# ===================================================================
# Best-response query (Definition 4.7.1, Lemma 4.7.2) against enumeration
# ===================================================================

BR_CASES = [
    (["..", ".."], 2, None, 0),
    (["...", "..."], 3, None, 1),
    (["...", "...", "..."], 2, None, 2),  # n does not divide |T|
    (["...", "...", "..."], 2, (2, 2), 3),  # n does not divide |T|, positive demands
    ([".....", "....."], 3, None, 4),  # n does not divide |T|
    (RUNNING, 3, (2, 2, 2), 5),
]


def _satisfying_play(territory: Territory, n: int, demands, seed: int) -> Play:
    rng = random.Random(seed)
    nb = territory.neighbor_map()
    d = demands or (0,) * n
    while True:
        play = random_play(len(territory), n, len(territory) // n, rng)
        r = rewards_from_owner(play.final_state, n, nb)
        if all(r[a] >= d[a] for a in range(n)):
            return play


def _follows_completion(run: Play, comp: CanonicalCompletion, deviator: int) -> bool:
    return all(
        run.actions[t][b] == comp.action(run.states[t], b)
        for t in range(run.horizon)
        for b in range(run.num_agents)
        if b != deviator
    )


class TestBestResponseOracle:
    @pytest.mark.parametrize("grid,n,demands,seed", BR_CASES)
    def test_best_response_matches_enumeration(self, grid, n, demands, seed):
        territory = Territory.from_ascii(grid)
        nb = territory.neighbor_map()
        m = len(territory) // n
        base = build_base_encoding(
            territory=territory, num_agents=n, horizon=m, demands=demands
        )
        d = demands or (0,) * n
        for k in range(2):
            play = _satisfying_play(territory, n, demands, 100 * seed + k)
            comp = CanonicalCompletion(play)
            current = rewards_from_owner(play.final_state, n, nb)
            for a in range(n):
                brute, _, _ = best_deviation(comp, a, nb, demands)
                sol = best_response_from_base(base, play, a)
                assert (
                    sol.is_sat
                    and sol.play is not None
                    and sol.payoff_by_agent is not None
                )
                # Lemma 4.7.2(3): the optimum is BR_a(pi), on the play's final state.
                assert sol.payoff_by_agent[a] == brute
                assert sol.encoded_reward_by_agent is not None
                assert sol.encoded_reward_by_agent[a] == brute
                # Lemma 4.7.2(1): the play is an admissible deviation.
                assert _follows_completion(sol.play, comp, a)
                assert all(sol.payoff_by_agent[b] >= d[b] for b in range(n))
                # Feasibility filter and VerifyNE query.
                imp = check_improvement_from_base(
                    base, agent=a, threshold=current[a], opponent_play=play
                )
                assert imp.is_sat == (brute > current[a])
                # S-IBIS binary search (Theorem 4.8.1).
                bs = best_response_binsearch_from_base(
                    base, play, a, current_reward=current[a]
                )
                if brute == current[a]:
                    assert bs.reason == "no_improvement" and bs.checks == 1
                else:
                    assert bs.reason == "improved" and bs.solution is not None
                    assert bs.solution.payoff_by_agent is not None
                    assert bs.solution.payoff_by_agent[a] == brute
                    assert bs.solution.play is not None
                    assert _follows_completion(bs.solution.play, comp, a)

    def test_example_4_8_1_four_checks(self):
        """Example 4.8.1: from the play of Figure 3.1 S-IBIS poses four checks, IP = (1, 0, 0)."""
        territory = Territory.from_ascii(RUNNING)
        fig = {0: [11, 10, 5, 1], 1: [8, 7, 12, 9], 2: [3, 2, 6, 4]}
        play = Play.from_actions(
            12, 3, [[fig[a][r] - 1 for a in range(3)] for r in range(4)]
        )
        base = build_base_encoding(
            territory=territory, num_agents=3, horizon=4, demands=(2, 2, 2)
        )
        current = base.rewards_of(play.final_state)
        assert current == (2, 4, 4)
        results = [
            best_response_binsearch_from_base(base, play, a, current_reward=current[a])
            for a in range(3)
        ]
        assert [r.reason for r in results] == [
            "improved",
            "no_improvement",
            "no_improvement",
        ]
        assert (
            results[0].solution is not None
            and results[0].solution.payoff_by_agent is not None
        )
        assert results[0].solution.payoff_by_agent[0] == 3
        assert sum(r.checks for r in results) == 4


# ===================================================================
# Z3 unknown is reported, never read as unsat (Section 7.1)
# ===================================================================


class TestUnknown:
    @pytest.fixture
    def always_unknown(self, monkeypatch):
        monkeypatch.setattr(smt_solve, "_check", lambda solver, *a: z3.unknown)

    def test_queries_report_unknown(self, always_unknown):
        territory = Territory.from_ascii(["..", ".."])
        base = build_base_encoding(territory=territory, num_agents=2, horizon=2)
        play = Play.from_actions(4, 2, [[0, 1], [2, 3]])
        assert find_feasible_from_base(base).reason == "unknown"
        assert best_response_from_base(base, play, 0).reason == "unknown"
        assert (
            check_improvement_from_base(
                base, agent=0, threshold=0, opponent_play=play
            ).reason
            == "unknown"
        )
        assert (
            best_response_binsearch_from_base(base, play, 0, current_reward=1).reason
            == "unknown"
        )

    def test_unknown_during_binary_search(self, monkeypatch):
        """A probe answering unknown stops the search with reason unknown."""
        territory = Territory.from_ascii(["...", "...", "..."])
        base = build_base_encoding(territory=territory, num_agents=2, horizon=4)
        play = Play.from_actions(9, 2, [[0, 8], [2, 6], [4, 1], [7, 3]])
        real = smt_solve._check
        calls = {"n": 0}

        def second_unknown(solver, *a):
            calls["n"] += 1
            return z3.unknown if calls["n"] == 2 else real(solver, *a)

        monkeypatch.setattr(smt_solve, "_check", second_unknown)
        current = base.rewards_of(play.final_state)
        res = best_response_binsearch_from_base(
            base, play, 0, current_reward=current[0]
        )
        assert res.reason == "unknown" and res.checks == 2

    def test_real_timeout_gives_unknown(self):
        """A 1 ms timeout on a 6x6 optimisation query is answered with unknown."""
        territory = Territory.from_ascii(["......"] * 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_collective_optimality(
                territory=territory, num_agents=2, horizon=18, timeout_ms=1
            )
        assert sol.reason in ("unknown", "sat")
        if not sol.is_sat:
            assert sol.reason == "unknown"


class TestSymmetryScope:
    def test_symmetry_ignored_for_individual_objective(self):
        """Lex-leader relabels agents, so it is not added to a single-agent query."""
        territory = Territory.from_ascii(["...", "...", "..."])
        with pytest.warns(UserWarning, match="symmetry_breaking ignored"):
            sol = solve_smt_game(
                territory=territory,
                num_agents=2,
                horizon=4,
                objective=1,
                symmetry_breaking=True,
            )
        assert sol.payoff_by_agent is not None and sol.payoff_by_agent[1] == 4

    def test_collective_optimum_unchanged_by_symmetry(self):
        territory = Territory.from_ascii(["..#", "...", ".#."])
        plain = solve_collective_optimality(
            territory=territory, num_agents=2, horizon=3
        )
        sym = solve_collective_optimality(
            territory=territory, num_agents=2, horizon=3, symmetry_breaking=True
        )
        assert plain.payoff_by_agent is not None and sym.payoff_by_agent is not None
        assert sum(plain.payoff_by_agent) == sum(sym.payoff_by_agent)
