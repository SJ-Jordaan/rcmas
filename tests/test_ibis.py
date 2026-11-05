"""Tests for ibis.py (thesis Algorithms 1 and 2).

- Example 4.7.1 and Remark 4.7.1 are reproduced exactly by the loop of
  Algorithm 1 with a brute-force best response that returns the
  lexicographically smallest optimal claim sequence, as the example states.
- Every equilibrium that IBIS or S-IBIS returns on a tiny instance is checked
  by exhaustive enumeration of the deviations against its canonical
  completion (Theorem 4.7.3, Corollary 4.8.2).
"""

from __future__ import annotations

import itertools

import pytest

import rcmas.smt_solve as smt_solve
from rcmas.ibis import BestResponse, IbisResult, run_ibis_loop, solve_ibis, solve_sibis
from rcmas.model import Territory, rewards_from_owner
from rcmas.strategy import CanonicalCompletion, Play

from .brute_force import best_deviation, is_equilibrium, max_welfare
from .replay import QueryReplay, disable_skip

z3 = pytest.importorskip("z3")

RUNNING = ["..#.", "#...", ".#.#", "...#", "#.##"]
FIG_3_1 = {
    0: [11, 10, 5, 1],
    1: [8, 7, 12, 9],
    2: [3, 2, 6, 4],
}  # thesis sector numbers


def _fig_3_1_play() -> Play:
    return Play.from_actions(
        12, 3, [[FIG_3_1[a][r] - 1 for a in range(3)] for r in range(4)]
    )


def _thesis_claims(play: Play) -> dict[int, list[int]]:
    return {a: [i + 1 for i in play.claims_of(a)] for a in range(play.num_agents)}


def _lex_best_response(territory: Territory, demands):
    nb = territory.neighbor_map()

    def br(play: Play, agent: int, current: tuple[int, ...]) -> BestResponse:
        value, _, run = best_deviation(CanonicalCompletion(play), agent, nb, demands)
        return BestResponse("ok", value, run, None, 0)

    return br


# ===================================================================
# Example 4.7.1 and Remark 4.7.1
# ===================================================================


class TestExample471:
    def _run(self, demands, max_iters=30) -> IbisResult:
        t = Territory.from_ascii(RUNNING)
        nb = t.neighbor_map()
        return run_ibis_loop(
            initial_play=_fig_3_1_play(),
            initial_solution=None,
            best_response=_lex_best_response(t, demands),
            rewards_of=lambda s: rewards_from_owner(s, 3, nb),
            max_iters=max_iters,
            territory=t,
        )

    def test_walkthrough_table(self):
        res = self._run((2, 2, 2))
        table = [(it.index, it.rewards, it.improvement, it.chosen) for it in res.trace]
        assert table == [
            (0, (2, 4, 4), (1, 0, 0), 0),
            (1, (3, 2, 3), (0, 1, 0), 1),
            (2, (2, 3, 3), (1, 0, 0), 0),
            (3, (3, 2, 3), (0, 2, 0), 1),
            (4, (2, 4, 2), (1, 0, 1), 0),
            (5, (3, 3, 2), (0, 0, 1), 2),
            (6, (3, 3, 3), (0, 0, 0), None),
        ]
        assert res.outcome == "ne" and res.found_ne and res.iterations == 6
        assert res.rewards == (3, 3, 3)
        assert _thesis_claims(res.play) == {
            0: [7, 10, 11, 5],
            1: [8, 6, 2, 9],
            2: [3, 4, 1, 12],
        }

    def test_iteration_0_update(self):
        """In iteration 0, a claims 7, 10, 11, 1; b claims 8, 2, 5, 9; c claims 3, 4, 6, 12."""
        res = self._run((2, 2, 2), max_iters=0)
        assert res.reason == "max_iters" and res.iterations == 0
        assert _thesis_claims(res.play) == {
            0: [7, 10, 11, 1],
            1: [8, 2, 5, 9],
            2: [3, 4, 6, 12],
        }

    def test_remark_4_7_1_demand_zero_recurs(self):
        """With demands 0 the run returns in iteration 6 to the play of iteration 2."""
        res = self._run((0, 0, 0))
        assert res.outcome == "n/c" and res.reason == "cycle"
        assert res.iterations == 6
        assert res.trace[6].rewards == res.trace[2].rewards
        assert [it.chosen for it in res.trace[:6]] == [0, 1, 0, 1, 0, 1]

    def test_smt_ibis_iteration_0(self):
        """IBIS on Z3 from the play of Figure 3.1: IP^0 = (1, 0, 0), a updates."""
        t = Territory.from_ascii(RUNNING)
        res = solve_ibis(
            territory=t,
            num_agents=3,
            demands=(2, 2, 2),
            initial_play=_fig_3_1_play(),
            max_iters=0,
        )
        assert res.trace[0].rewards == (2, 4, 4)
        assert res.trace[0].improvement == (1, 0, 0)
        assert res.trace[0].chosen == 0
        assert res.rewards is not None and res.rewards[0] == 3

    def test_smt_sibis_iteration_0_four_checks(self, monkeypatch):
        """Example 4.8.1: four satisfiability checks in iteration 0.

        Agents b and c have reward m = 4, so the tool poses no query for
        them (``at_reward_bound``): two checks for a.  With the skip
        disabled it poses the four checks of the example, the filters of b
        and c among them, which are unsatisfiable.
        """
        t = Territory.from_ascii(RUNNING)

        def run():
            return solve_sibis(
                territory=t,
                num_agents=3,
                demands=(2, 2, 2),
                initial_play=_fig_3_1_play(),
                max_iters=0,
            )

        res = run()
        assert res.trace[0].rewards == (2, 4, 4)
        assert res.trace[0].improvement == (1, 0, 0)
        assert res.trace[0].checks == 2
        disable_skip(monkeypatch)
        full = run()
        assert full.trace[0].improvement == (1, 0, 0)
        assert full.trace[0].checks == 4


# ===================================================================
# The loop: selection, recurrence, bound
# ===================================================================


def _line_plays(count: int) -> list[Play]:
    """Distinct plays of a 1x6 line with 3 agents (m = 2)."""
    plays = []
    for perm in itertools.permutations(range(6)):
        plays.append(Play.from_actions(6, 3, [perm[:3], perm[3:]]))
        if len(plays) == count:
            break
    return plays


class TestLoop:
    def test_selects_largest_absolute_improvement(self):
        """Rewards (1, 4, 4) and gains (1, 2, 2): agent 1 (largest gain, smallest index)."""
        plays = _line_plays(4)
        rewards = {plays[0]: (1, 4, 4)}
        gain = {plays[0]: (1, 2, 2)}

        def br(play, a, cur):
            g = gain.get(play, (0, 0, 0))[a]
            return BestResponse("ok", cur[a] + g, plays[1 + a] if g else None, None, 1)

        res = run_ibis_loop(
            initial_play=plays[0],
            initial_solution=None,
            best_response=br,
            rewards_of=lambda s: rewards.get(
                next(p for p in plays if p.final_state == s), (0, 0, 0)
            ),
            max_iters=5,
        )
        assert res.trace[0].chosen == 1
        assert res.play == plays[2]
        assert res.outcome == "ne" and res.iterations == 1

    def test_recurrence_detected(self):
        plays = _line_plays(2)

        def br(play, a, cur):
            other = plays[1] if play == plays[0] else plays[0]
            return BestResponse("ok", cur[a] + (1 if a == 0 else 0), other, None, 1)

        res = run_ibis_loop(
            initial_play=plays[0],
            initial_solution=None,
            best_response=br,
            rewards_of=lambda s: (0, 0, 0),
            max_iters=10,
        )
        assert res.outcome == "n/c" and res.reason == "cycle" and res.iterations == 2

    def test_iteration_bound(self):
        """The loop runs for i = 0, ..., N and then reports n/c."""
        plays = _line_plays(10)
        nxt = {plays[k]: plays[k + 1] for k in range(9)}

        def br(play, a, cur):
            return BestResponse(
                "ok", cur[a] + (a == 0), nxt[play] if a == 0 else None, None, 1
            )

        res = run_ibis_loop(
            initial_play=plays[0],
            initial_solution=None,
            best_response=br,
            rewards_of=lambda s: (0, 0, 0),
            max_iters=3,
        )
        assert res.outcome == "n/c" and res.reason == "max_iters"
        assert res.iterations == 3 and len(res.trace) == 4
        assert res.play == plays[4]

    def test_unknown_stops_run(self):
        plays = _line_plays(1)

        def br(play, a, cur):
            return BestResponse("unknown" if a == 1 else "ok", cur[a], None, None, 1)

        res = run_ibis_loop(
            initial_play=plays[0],
            initial_solution=None,
            best_response=br,
            rewards_of=lambda s: (0, 0, 0),
            max_iters=3,
        )
        assert res.outcome == "unk" and res.reason == "unknown" and not res.found_ne
        assert res.play == plays[0]

    def test_negative_potential_is_an_error(self):
        plays = _line_plays(1)
        with pytest.raises(RuntimeError, match="negative"):
            run_ibis_loop(
                initial_play=plays[0],
                initial_solution=None,
                best_response=lambda p, a, cur: BestResponse(
                    "ok", cur[a] - 1, None, None, 1
                ),
                rewards_of=lambda s: (1, 1, 1),
                max_iters=1,
            )


# ===================================================================
# IBIS and S-IBIS on Z3: soundness against enumeration
# ===================================================================

SOUNDNESS_CASES = [
    (["..", ".."], 2, None),
    (["...", "..."], 3, None),
    (["...", "...", "..."], 2, None),  # n does not divide |T|
    ([".....", "....."], 3, None),  # n does not divide |T|
    (["...", "...", "..."], 3, (2, 2, 2)),
    (["..#", "...", ".#."], 2, (1, 1)),
]


class TestSoundness:
    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    @pytest.mark.parametrize("grid,n,demands", SOUNDNESS_CASES)
    def test_returned_equilibria_are_equilibria(self, solver, grid, n, demands):
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        res = solver(territory=t, num_agents=n, demands=demands)
        assert res.outcome in ("ne", "n/c")
        assert res.play is not None
        assert res.rewards == rewards_from_owner(res.play.final_state, n, nb)
        assert res.strategy == res.play.to_coords(t)
        assert res.final_solution is not None and res.final_solution.play == res.play
        d = demands or (0,) * n
        assert all(r >= x for r, x in zip(res.rewards, d))
        if res.outcome == "ne":
            assert is_equilibrium(res.play, nb, demands)
            assert res.trace[-1].improvement == (0,) * n
            comp = res.completion()
            assert comp is not None and comp.play == res.play

    def test_symmetry_and_plain_both_sound(self):
        t = Territory.from_ascii(["...", "...", "..."])
        for sym in (False, True):
            res = solve_sibis(territory=t, num_agents=3, symmetry=sym)
            assert res.outcome == "ne"
            assert is_equilibrium(res.play, t.neighbor_map())

    def test_weighted_abstract_game(self):
        """The loop runs on a weighted game with block adjacency (Section 6.1)."""
        abstract = Territory.from_ascii([".....", ""])
        w = (2, 1, 3, 1, 2)
        nb = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
        res = solve_sibis(
            territory=abstract, num_agents=2, weights=w, custom_neighbors=nb
        )
        assert res.play is not None
        assert res.rewards == rewards_from_owner(res.play.final_state, 2, nb, w)
        if res.outcome == "ne":
            assert is_equilibrium(res.play, nb, None, w)


class TestOutcomes:
    def test_no_satisfying_strategy(self):
        t = Territory.from_ascii(["...", "...", "..."])
        res = solve_ibis(territory=t, num_agents=3, demands=(4, 0, 0))
        assert res.outcome == "uns" and res.reason == "unsat" and not res.is_sat
        assert res.play is None

    def test_unknown_in_initial_query(self, monkeypatch):
        monkeypatch.setattr(smt_solve, "_check", lambda solver, *a: z3.unknown)
        t = Territory.from_ascii(["..", ".."])
        res = solve_sibis(territory=t, num_agents=2)
        assert res.outcome == "unk" and res.reason == "unknown"

    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    @pytest.mark.parametrize("call", [1, 2])
    def test_unknown_in_best_response_stops(self, monkeypatch, solver, call):
        """An unknown best-response check stops the run with ``unk``.

        The initial play is given, with rewards (1, 1) below m = 4, so that
        both agents are queried in iteration 0 (an agent at m is not) and
        the first two checks are best-response checks.
        """
        real = smt_solve._check
        calls = {"n": 0}

        def nth_unknown(s, *a):
            calls["n"] += 1
            return z3.unknown if calls["n"] == call else real(s, *a)

        monkeypatch.setattr(smt_solve, "_check", nth_unknown)
        t = Territory.from_ascii(["...", "...", "..."])
        play = Play.from_actions(9, 2, [[0, 1], [2, 3], [4, 5], [6, 7]])
        assert play.rewards(t.neighbor_map()) == (1, 1)
        res = solver(territory=t, num_agents=2, initial_play=play)
        assert res.outcome == "unk" and not res.found_ne
        assert res.play == play and res.iterations == 0 and calls["n"] == call

    def test_symmetry_only_in_initial_query(self, monkeypatch):
        calls = {"n": 0}
        real = smt_solve.symmetry_constraints

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(smt_solve, "symmetry_constraints", spy)
        t = Territory.from_ascii(["...", "...", "..."])
        res = solve_sibis(territory=t, num_agents=2, symmetry=True)
        assert res.iterations >= 0 and calls["n"] == 1
        solve_ibis(territory=t, num_agents=2, symmetry=False)
        assert calls["n"] == 1


# ===================================================================
# Input validation
# ===================================================================


class TestValidation:
    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    def test_zero_agents_raises(self, solver):
        with pytest.raises(ValueError, match="num_agents"):
            solver(territory=Territory.from_ascii([".."]), num_agents=0, horizon=1)

    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    def test_horizon_must_be_m(self, solver):
        with pytest.raises(ValueError, match="horizon"):
            solver(territory=Territory.from_ascii([".."]), num_agents=1, horizon=1)

    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    def test_negative_max_iters_raises(self, solver):
        with pytest.raises(ValueError, match="max_iters"):
            solver(territory=Territory.from_ascii([".."]), num_agents=1, max_iters=-1)

    def test_horizon_defaults_to_m(self):
        res = solve_sibis(territory=Territory.from_ascii(["...", "..."]), num_agents=2)
        assert res.play is not None and res.play.horizon == 3

    def test_trivial_one_round(self):
        res = solve_ibis(
            territory=Territory.from_ascii([".."]), num_agents=2, horizon=1
        )
        assert res.outcome == "ne" and res.rewards == (1, 1)

    def test_horizon_zero_game(self):
        """|T| < |Agt|: the empty play is an equilibrium (no agent acts)."""
        res = solve_sibis(territory=Territory.from_ascii(["."]), num_agents=2)
        assert res.outcome == "ne" and res.rewards == (0, 0)


# ===================================================================
# The protocol of Chapter 7: no iteration bound, collectively optimal start
# ===================================================================

PROTOCOL_CASES = SOUNDNESS_CASES + [
    (["..", ".", ".."], 1, None),  # L-shape
    (["..#.."], 2, None),  # disconnected
    (["...", "..."], 2, (2, 2)),
]


class TestUnboundedAndCollective:
    """``max_iters=None`` and ``initial="collective"`` (Sections 4.6 and 4.7)."""

    def test_unbounded_loop_stops_at_recurrence(self):
        """Without a bound the loop of Remark 4.7.1 still ends, by recurrence."""
        plays = _line_plays(2)

        def br(play, a, cur):
            other = plays[1] if play == plays[0] else plays[0]
            return BestResponse("ok", cur[a] + (1 if a == 0 else 0), other, None, 1)

        res = run_ibis_loop(
            initial_play=plays[0],
            initial_solution=None,
            best_response=br,
            rewards_of=lambda s: (0, 0, 0),
            max_iters=None,
        )
        assert res.outcome == "n/c" and res.reason == "cycle" and res.iterations == 2

    def test_unbounded_loop_runs_past_any_bound(self):
        """A chain of 40 improving plays converges without a bound, but not with N = 30."""
        plays = _line_plays(41)
        nxt = {plays[k]: plays[k + 1] for k in range(40)}

        def br(play, a, cur):
            step = a == 0 and play in nxt
            return BestResponse(
                "ok", cur[a] + step, nxt[play] if step else None, None, 1
            )

        kw = dict(
            initial_play=plays[0],
            initial_solution=None,
            best_response=br,
            rewards_of=lambda s: (0, 0, 0),
        )
        bounded = run_ibis_loop(**kw, max_iters=30)
        assert bounded.outcome == "n/c" and bounded.reason == "max_iters"
        free = run_ibis_loop(**kw, max_iters=None)
        assert free.outcome == "ne" and free.iterations == 40
        assert free.play == plays[40] and len(free.trace) == 41

    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    @pytest.mark.parametrize("initial", ["satisfying", "collective"])
    @pytest.mark.parametrize("symmetry", [False, True])
    @pytest.mark.parametrize("grid,n,demands", PROTOCOL_CASES)
    def test_unbounded_runs_terminate_and_are_sound(
        self, solver, initial, symmetry, grid, n, demands
    ):
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        res = solver(
            territory=t,
            num_agents=n,
            demands=demands,
            max_iters=None,
            initial=initial,
            symmetry=symmetry,
        )
        assert res.outcome in ("ne", "n/c")
        assert res.reason in ("ne", "cycle")  # no bound, so never "max_iters"
        assert res.initial == initial and res.initial_query_s is not None
        assert res.initial_play is not None
        assert res.initial_rewards == rewards_from_owner(
            res.initial_play.final_state, n, nb
        )
        assert res.trace[0].rewards == res.initial_rewards
        if res.outcome == "ne":
            assert is_equilibrium(res.play, nb, demands)
            assert res.returned_initial == (res.iterations == 0)
            if res.returned_initial:
                assert res.play == res.initial_play

    @pytest.mark.parametrize("symmetry", [False, True])
    @pytest.mark.parametrize("grid,n,demands", PROTOCOL_CASES)
    def test_collective_start_is_a_collectively_optimal_model_of_F(
        self, symmetry, grid, n, demands
    ):
        """``pi^0`` satisfies F and its welfare is the brute-force optimum of ``f``."""
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        m = len(t) // n
        res = solve_sibis(
            territory=t,
            num_agents=n,
            demands=demands,
            initial="collective",
            symmetry=symmetry,
            max_iters=0,
        )
        play0 = res.initial_play
        assert play0 is not None and play0.horizon == m
        r0 = rewards_from_owner(play0.final_state, n, nb)
        assert res.initial_rewards == r0
        d = demands or (0,) * n
        assert all(r0[a] >= d[a] for a in range(n))  # [phi_sat] of F
        assert sum(r0) == max_welfare(len(t), n, m, nb, demands)

    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    @pytest.mark.parametrize(
        "grid,n", [(["...", "..."], 2), (["...", "..."], 3), (["..#..", "..#.."], 2)]
    )
    def test_all_m_collective_play_returns_in_iteration_0(self, solver, grid, n):
        """Every reward m: an equilibrium under any completion, returned at once."""
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        m = len(t) // n
        assert max_welfare(len(t), n, m, nb) == n * m  # the optimum gives every agent m
        res = solver(territory=t, num_agents=n, initial="collective", max_iters=None)
        assert res.initial_rewards == (m,) * n
        assert res.outcome == "ne" and res.iterations == 0 and res.returned_initial
        assert len(res.trace) == 1 and res.trace[0].improvement == (0,) * n
        assert res.play == res.initial_play
        assert is_equilibrium(res.play, nb)

    def test_collective_start_in_a_weighted_game(self):
        """In an abstract game the start maximises the sum of weighted rewards."""
        abstract = Territory.from_ascii([".....", ""])
        w = (2, 1, 3, 1, 2)
        nb = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
        res = solve_sibis(
            territory=abstract,
            num_agents=2,
            weights=w,
            custom_neighbors=nb,
            initial="collective",
            max_iters=None,
        )
        r0 = rewards_from_owner(res.initial_play.final_state, 2, nb, w)
        assert res.initial_rewards == r0
        assert sum(r0) == max_welfare(5, 2, 2, nb, None, w)
        if res.outcome == "ne":
            assert is_equilibrium(res.play, nb, None, w)

    def test_unknown_collective_query_stops_with_unk(self, monkeypatch):
        """An initial query answered unknown (for example at the Z3 timeout) gives unk."""
        monkeypatch.setattr(smt_solve, "_check", lambda solver, *a: z3.unknown)
        seen = []
        res = solve_ibis(
            territory=Territory.from_ascii(["..", ".."]),
            num_agents=2,
            initial="collective",
            max_iters=None,
            on_initial=seen.append,
        )
        assert res.outcome == "unk" and res.reason == "unknown" and not res.is_sat
        assert res.initial == "collective" and res.initial_query_s is not None
        assert res.initial_play is None and not res.returned_initial
        assert len(seen) == 1 and seen[0].reason == "unknown"

    def test_collective_query_uses_the_timeout(self, monkeypatch):
        """The initial query gets the run's Z3 timeout, like every other query."""
        timeouts = []
        real = smt_solve._new_solver

        def spy(kind, timeout_ms):
            timeouts.append((kind, timeout_ms))
            return real(kind, timeout_ms)

        monkeypatch.setattr(smt_solve, "_new_solver", spy)
        solve_sibis(
            territory=Territory.from_ascii(["...", "..."]),
            num_agents=2,
            initial="collective",
            timeout_ms=12345,
        )
        assert timeouts[0] == ("opt", 12345)  # the collective query is an optimisation
        assert all(t == 12345 for _, t in timeouts)

    def test_on_initial_reports_the_initial_query(self):
        seen = []
        res = solve_sibis(
            territory=Territory.from_ascii(["...", "..."]),
            num_agents=2,
            initial="collective",
            on_initial=seen.append,
        )
        (init,) = seen
        assert init.reason == "sat" and init.initial == "collective"
        assert init.play == res.initial_play and init.query_s == res.initial_query_s

    def test_given_initial_play_has_no_query(self):
        play = Play.from_actions(4, 2, [[0, 3], [1, 2]])
        res = solve_ibis(
            territory=Territory.from_ascii(["..", ".."]),
            num_agents=2,
            initial_play=play,
            max_iters=None,
        )
        assert res.initial == "given" and res.initial_query_s is None
        assert res.initial_play == play

    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    def test_invalid_initial_query_raises(self, solver):
        with pytest.raises(ValueError, match="initial"):
            solver(territory=Territory.from_ascii([".."]), num_agents=1, initial="best")


@pytest.mark.parametrize(
    "grid,n",
    [
        ("local_symmetry/L-4x4-2.txt", 2),  # local symmetry constraints
        ("disconnected/2x2-2x2.txt", 4),  # component-wise constraints
        ("disconnected/3x2-2x1.txt", 2),
        ("asymmetric/interesting.txt", 3),  # the optimum does not give every agent m
    ],
)
def test_collective_start_on_benchmark_grids(grid, n):
    """With and without the symmetry constraints, ``pi^0`` reaches the optimum of ``f``.

    The constraints of Chapter 5 keep a collectively optimal model (Section
    5.4.5), so the welfare of the initial play is the same in both settings.
    """
    import os

    from experiments.scenarios import GRIDS_DIR

    with open(os.path.join(GRIDS_DIR, grid), encoding="utf-8") as f:
        t = Territory.from_ascii(f)
    nb = t.neighbor_map()
    m = len(t) // n
    best = max_welfare(len(t), n, m, nb)
    for symmetry in (False, True):
        res = solve_sibis(
            territory=t,
            num_agents=n,
            initial="collective",
            symmetry=symmetry,
            max_iters=0,
        )
        assert res.initial_rewards is not None
        assert sum(res.initial_rewards) == best, (grid, n, symmetry)


# ===================================================================
# No best-response query for an agent at R_max (at_reward_bound)
# ===================================================================


def _count_checks(monkeypatch) -> list:
    """Record every solver check (every query of a run goes through smt_solve._check)."""
    calls: list = []
    real = smt_solve._check

    def counting(solver, *a):
        calls.append(solver)
        return real(solver, *a)

    monkeypatch.setattr(smt_solve, "_check", counting)
    return calls


ALL_M_CASES = [
    (["...", "..."], 2),
    (["...", "..."], 3),
    (["..#..", "..#.."], 2),
    (["....", "...."], 2),
    (["....", "...."], 4),
]


class TestRewardBound:
    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    @pytest.mark.parametrize("grid,n", ALL_M_CASES)
    @pytest.mark.parametrize("symmetry", [False, True])
    def test_all_m_start_poses_no_best_response_query(
        self, monkeypatch, solver, grid, n, symmetry
    ):
        """From a collectively optimal play with every reward m, no query but smt(F, f)."""
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        m = len(t) // n
        assert max_welfare(len(t), n, m, nb) == n * m
        checks = _count_checks(monkeypatch)
        res = solver(
            territory=t,
            num_agents=n,
            initial="collective",
            max_iters=None,
            symmetry=symmetry,
        )
        assert res.initial_rewards == (m,) * n
        assert res.outcome == "ne" and res.iterations == 0 and res.returned_initial
        assert res.trace[0].improvement == (0,) * n and res.trace[0].checks == 0
        assert len(checks) == 1  # the initial query smt(F, f) and nothing else
        assert is_equilibrium(res.play, nb)

    def test_best_response_functions_skip_only_at_the_bound(self, monkeypatch):
        """The best responses answer at R_max without a query, below it with one."""
        from rcmas.ibis import binsearch_best_response, smt_best_response
        from rcmas.smt_solve import build_base_encoding

        t = Territory.from_ascii(["...", "..."])
        base = build_base_encoding(territory=t, num_agents=3, horizon=2)
        play = Play.from_actions(6, 3, [[0, 1, 3], [4, 2, 5]])
        current = play.rewards(t.neighbor_map())
        assert current == (1, 2, 1) and base.max_reward == 2
        for make in (smt_best_response, binsearch_best_response):
            br = make(base)
            checks = _count_checks(monkeypatch)
            at = br(play, 1, current)
            assert (at.status, at.reward, at.play, at.checks) == ("ok", 2, None, 0)
            assert checks == []
            below = br(play, 0, current)
            assert below.status == "ok" and below.checks >= 1 and len(checks) >= 1
            assert (
                below.reward
                == best_deviation(CanonicalCompletion(play), 0, t.neighbor_map(), None)[
                    0
                ]
            )

    @pytest.mark.parametrize("solver", [solve_ibis, solve_sibis])
    @pytest.mark.parametrize(
        "grid,n,demands,start",
        [
            (RUNNING, 3, (2, 2, 2), "fig_3_1"),
            (["....", "...."], 4, None, "satisfying"),
            (["..#", "...", ".#."], 2, (1, 1), "satisfying"),
            ([".....", "....."], 3, None, "satisfying"),
            (["....", "...", "..#."], 3, None, "satisfying"),
            (RUNNING, 3, None, "collective"),
        ],
    )
    def test_same_run_with_and_without_the_skip(
        self, monkeypatch, solver, grid, n, demands, start
    ):
        """The skip removes queries only: same plays, potentials and outcome.

        The run without the skip poses every query and the run with it is
        answered from that record (``tests/replay.py``), since Z3's models
        depend on process-wide state.  Every query the skip leaves out is
        that of an agent at R_max, with potential 0.
        """
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        m = len(t) // n
        kw = dict(territory=t, num_agents=n, demands=demands, max_iters=None)
        if start == "fig_3_1":
            kw["initial_play"] = _fig_3_1_play()
        else:
            kw["initial"] = start
        replay = QueryReplay(monkeypatch)
        with monkeypatch.context() as mp:
            disable_skip(mp)
            full = solver(**kw)
        recorded = replay.replay()
        fast = solver(**kw)
        assert (fast.outcome, fast.reason, fast.iterations) == (
            full.outcome,
            full.reason,
            full.iterations,
        )
        assert fast.play == full.play and fast.rewards == full.rewards
        assert fast.initial_play == full.initial_play
        assert [
            (it.index, it.rewards, it.improvement, it.chosen) for it in fast.trace
        ] == [(it.index, it.rewards, it.improvement, it.chosen) for it in full.trace]
        # The left-out queries: one per (iteration, agent at m), each with IP 0.
        at_bound = [
            (it.index, a)
            for it in full.trace
            for a in range(n)
            if it.improvement and it.rewards[a] == m
        ]
        assert at_bound, "the case must exercise the skip"
        assert all(full.trace[i].improvement[a] == 0 for i, a in at_bound)
        assert len(replay.best_responses(recorded)) - len(
            replay.best_responses()
        ) == len(at_bound)
        assert sum(it.checks for it in full.trace) - sum(
            it.checks for it in fast.trace
        ) == len(at_bound)
        if fast.outcome == "ne":
            assert is_equilibrium(fast.play, nb, demands)
