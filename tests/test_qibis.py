"""Tests for qibis.py (thesis Algorithm 3, Section 4.9).

- The loop of Algorithm 3, driven by brute-force best responses, reproduces
  Example 4.7.1 through its certification round when the candidate sets
  hold only the current actions, and reproduces it through its restricted
  queries when they hold every action.
- The restricted query on Z3 is checked against exhaustive enumeration of
  the deviations that respect the candidate sets, and its models let the
  opponents follow the canonical completion (Remark 4.9.1).
- Every equilibrium that Q-IBIS returns on a tiny instance is checked by
  enumeration of all deviations against its canonical completion
  (Corollary 4.9.1).
"""

from __future__ import annotations

import itertools
import random

import pytest

import rcmas.smt_solve as smt_solve
from rcmas.ibis import BestResponse, solve_ibis
from rcmas.model import Territory, rewards_from_owner
from rcmas.qibis import (
    QibisConfig,
    QibisLearner,
    QibisResult,
    SectorCandidates,
    restricted_smt_best_response,
    run_qibis_loop,
    solve_qibis,
)
from rcmas.qlearning import LearningGame, LearningParams
from rcmas.smt_solve import build_base_encoding
from rcmas.strategy import CanonicalCompletion, Play, is_live

from .brute_force import (
    best_deviation,
    deviations,
    is_equilibrium,
    max_welfare,
    random_play,
)
from .replay import QueryReplay, disable_skip

z3 = pytest.importorskip("z3")

RUNNING = ["..#.", "#...", ".#.#", "...#", "#.##"]
FIG_3_1 = {
    0: [11, 10, 5, 1],
    1: [8, 7, 12, 9],
    2: [3, 2, 6, 4],
}  # thesis sector numbers

EXAMPLE_4_7_1 = [
    (0, (2, 4, 4), (1, 0, 0), 0),
    (1, (3, 2, 3), (0, 1, 0), 1),
    (2, (2, 3, 3), (1, 0, 0), 0),
    (3, (3, 2, 3), (0, 2, 0), 1),
    (4, (2, 4, 2), (1, 0, 1), 0),
    (5, (3, 3, 2), (0, 0, 1), 2),
    (6, (3, 3, 3), (0, 0, 0), None),
]

FAST = dict(offline_episodes=30, online_episodes=30)


def _fig_3_1_play() -> Play:
    return Play.from_actions(
        12, 3, [[FIG_3_1[a][r] - 1 for a in range(3)] for r in range(4)]
    )


def _thesis_claims(play: Play) -> dict[int, list[int]]:
    return {a: [i + 1 for i in play.claims_of(a)] for a in range(play.num_agents)}


def _respects(run: Play, seq: tuple[int, ...], cand: SectorCandidates) -> bool:
    """True if the deviator's claims *seq* lie in *cand* at every candidate state of *run*."""
    return all(
        run.states[t] not in cand or seq[t] in cand[run.states[t]]
        for t in range(run.horizon)
    )


def _restricted_best(
    play: Play, agent: int, nb, cand: SectorCandidates, demands=None, weights=None
) -> tuple[int, tuple[int, ...], Play]:
    """The restricted best response by enumeration: the lexicographically smallest optimum."""
    comp = CanonicalCompletion(play)
    n = play.num_agents
    d = demands if demands is not None else (0,) * n
    best = None
    for seq, run in deviations(
        comp.action, agent, len(play.states[0]), n, play.horizon
    ):
        if not _respects(run, seq, cand):
            continue
        r = rewards_from_owner(run.final_state, n, nb, weights)
        if any(r[b] < d[b] for b in range(n)):
            continue
        if best is None or r[agent] > best[0]:
            best = (r[agent], seq, run)
    assert best is not None, "the play itself respects its candidate sets"
    return best


def _current_only(play: Play) -> list[SectorCandidates]:
    """Candidate sets with k = 0: only the current action at every state of the play."""
    return [
        {play.states[t]: (play.actions[t][a],) for t in range(play.horizon)}
        for a in range(play.num_agents)
    ]


def _all_actions(play: Play) -> list[SectorCandidates]:
    """Candidate sets that allow every unoccupied sector at the states of the play."""
    out = []
    for _ in range(play.num_agents):
        out.append(
            {
                play.states[t]: tuple(
                    i for i, o in enumerate(play.states[t]) if o == -1
                )
                for t in range(play.horizon)
            }
        )
    return out


def _brute_loop(refine, demands, max_iters=30) -> QibisResult:
    t = Territory.from_ascii(RUNNING)
    nb = t.neighbor_map()

    def restricted(play, a, cur, cand):
        value, _, run = _restricted_best(play, a, nb, cand, demands)
        return BestResponse("ok", value, run, None, 0)

    def unrestricted(play, a, cur):
        value, _, run = best_deviation(CanonicalCompletion(play), a, nb, demands)
        return BestResponse("ok", value, run, None, 0)

    return run_qibis_loop(
        initial_play=_fig_3_1_play(),
        initial_solution=None,
        refine=refine,
        restricted_best_response=restricted,
        best_response=unrestricted,
        rewards_of=lambda s: rewards_from_owner(s, 3, nb),
        max_iters=max_iters,
        territory=t,
    )


# ===================================================================
# The loop of Algorithm 3 on the running example
# ===================================================================


class TestLoopOnExample471:
    def test_certification_round_reproduces_ibis(self):
        """With only current actions as candidates every restricted potential is 0,
        and the certification round reproduces the run of Example 4.7.1."""
        res = _brute_loop(_current_only, (2, 2, 2))
        table = [(it.index, it.rewards, it.certified, it.chosen) for it in res.trace]
        assert table == EXAMPLE_4_7_1
        assert all(it.restricted == (0, 0, 0) for it in res.trace)
        assert res.outcome == "ne" and res.iterations == 6
        assert _thesis_claims(res.play) == {
            0: [7, 10, 11, 5],
            1: [8, 6, 2, 9],
            2: [3, 4, 1, 12],
        }

    def test_unrestricted_candidates_skip_certification_until_the_end(self):
        """With every action a candidate the restricted queries find the potentials
        of Example 4.7.1, and the certification round is posed only in iteration 6."""
        res = _brute_loop(_all_actions, (2, 2, 2))
        table = [(it.index, it.rewards, it.restricted, it.chosen) for it in res.trace]
        assert table == EXAMPLE_4_7_1
        assert [it.certified for it in res.trace] == [()] * 6 + [(0, 0, 0)]
        assert res.outcome == "ne"

    def test_no_return_without_certification(self):
        """Iteration 0: restricted potentials (0, 0, 0), but a can improve, so no return."""
        res = _brute_loop(_current_only, (2, 2, 2), max_iters=0)
        it = res.trace[0]
        assert it.restricted == (0, 0, 0) and it.certified == (1, 0, 0)
        assert it.chosen == 0 and it.improvement == (1, 0, 0)
        assert res.outcome == "n/c" and res.reason == "max_iters"
        assert _thesis_claims(res.play)[0] == [7, 10, 11, 1]

    def test_remark_4_7_1_recurrence(self):
        """With demands 0 the run returns in iteration 6 to the play of iteration 2."""
        res = _brute_loop(_current_only, (0, 0, 0))
        assert res.outcome == "n/c" and res.reason == "cycle" and res.iterations == 6


def _line_plays(count: int) -> list[Play]:
    plays = []
    for perm in itertools.permutations(range(6)):
        plays.append(Play.from_actions(6, 3, [perm[:3], perm[3:]]))
        if len(plays) == count:
            break
    return plays


class TestLoop:
    def _run(self, restricted, unrestricted, max_iters=5, plays=None):
        plays = plays or _line_plays(1)
        return run_qibis_loop(
            initial_play=plays[0],
            initial_solution=None,
            refine=lambda p: [{} for _ in range(3)],
            restricted_best_response=restricted,
            best_response=unrestricted,
            rewards_of=lambda s: (1, 1, 1),
            max_iters=max_iters,
        )

    def test_restricted_improvement_skips_certification(self):
        plays = _line_plays(3)
        calls = {"cert": 0}

        def restricted(play, a, cur, cand):
            if play == plays[0] and a == 2:
                return BestResponse("ok", cur[a] + 1, plays[1], None, 1)
            return BestResponse("ok", cur[a], None, None, 1)

        def unrestricted(play, a, cur):
            calls["cert"] += 1
            return BestResponse("ok", cur[a], None, None, 1)

        res = self._run(restricted, unrestricted, plays=plays)
        assert res.trace[0].chosen == 2 and res.trace[0].certified == ()
        assert res.outcome == "ne" and res.play == plays[1] and res.iterations == 1
        assert calls["cert"] == 3

    def test_unknown_in_restricted_query_stops(self):
        res = self._run(
            lambda p, a, cur, c: BestResponse(
                "unknown" if a == 1 else "ok", cur[a], None, None, 1
            ),
            lambda p, a, cur: pytest.fail("no certification after unknown"),
        )
        assert res.outcome == "unk" and res.reason == "unknown" and not res.found_ne
        assert res.trace[0].restricted == (0,)

    def test_unknown_in_certification_stops(self):
        res = self._run(
            lambda p, a, cur, c: BestResponse("ok", cur[a], None, None, 1),
            lambda p, a, cur: BestResponse(
                "unknown" if a == 2 else "ok", cur[a], None, None, 1
            ),
        )
        assert res.outcome == "unk" and res.trace[0].certified == (0, 0)

    @pytest.mark.parametrize("where", ["restricted", "certification"])
    def test_negative_potential_is_an_error(self, where):
        bad = BestResponse("ok", 0, None, None, 1)
        good = BestResponse("ok", 1, None, None, 1)
        with pytest.raises(RuntimeError, match="negative"):
            self._run(
                lambda p, a, cur, c: bad if where == "restricted" else good,
                lambda p, a, cur: bad,
            )

    def test_iteration_bound_and_recurrence(self):
        plays = _line_plays(10)
        nxt = {plays[k]: plays[k + 1] for k in range(9)}

        def restricted(play, a, cur, cand):
            return BestResponse(
                "ok", cur[a] + (a == 0), nxt[play] if a == 0 else None, None, 1
            )

        res = self._run(restricted, None, max_iters=3, plays=plays)
        assert res.reason == "max_iters" and res.iterations == 3
        assert len(res.trace) == 4 and res.play == plays[4]

        back = {plays[0]: plays[1], plays[1]: plays[0]}
        res = self._run(
            lambda p, a, cur, c: BestResponse(
                "ok", cur[a] + (a == 0), back[p] if a == 0 else None, None, 1
            ),
            None,
            max_iters=10,
            plays=plays,
        )
        assert res.reason == "cycle" and res.outcome == "n/c" and res.iterations == 2


# ===================================================================
# The restricted query on Z3 (line 14, Remark 4.9.1)
# ===================================================================


def _random_candidates(play: Play, agent: int, rng: random.Random) -> SectorCandidates:
    """Random candidate sets: the play's states with the current action, plus visited states."""
    n = play.num_agents
    S = len(play.states[0])
    comp = CanonicalCompletion(play)
    cand: SectorCandidates = {}
    for t in range(play.horizon):
        s = play.states[t]
        free = [i for i in range(S) if s[i] == -1]
        extra = rng.sample(free, rng.randint(0, min(2, len(free))))
        cand[s] = tuple(sorted({play.actions[t][agent], *extra}))
    runs = [run for _, run in deviations(comp.action, agent, S, n, play.horizon)]
    for run in rng.sample(runs, min(6, len(runs))):
        for t in range(1, run.horizon):
            s = run.states[t]
            if s in cand or rng.random() < 0.5:
                continue
            free = [i for i in range(S) if s[i] == -1]
            cand[s] = tuple(sorted(rng.sample(free, rng.randint(0, min(2, len(free))))))
    return cand


RESTRICTED_CASES = [
    (["...", "..."], 2, None, 0),
    (["...", "..."], 3, None, 1),
    (["...", "...", "..."], 2, None, 2),  # n does not divide |T|
    (["...", "...", "..."], 2, (2, 2), 3),
]


class TestRestrictedQuery:
    @pytest.mark.parametrize("grid,n,demands,seed", RESTRICTED_CASES)
    def test_matches_enumeration(self, grid, n, demands, seed):
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        m = len(t) // n
        base = build_base_encoding(
            territory=t, num_agents=n, horizon=m, demands=demands
        )
        br = restricted_smt_best_response(base)
        rng = random.Random(seed)
        d = demands or (0,) * n
        play = None
        while play is None:
            cand_play = random_play(len(t), n, m, rng)
            r = rewards_from_owner(cand_play.final_state, n, nb)
            if all(r[a] >= d[a] for a in range(n)):
                play = cand_play
        comp = CanonicalCompletion(play)
        current = rewards_from_owner(play.final_state, n, nb)
        for a in range(n):
            cand = _random_candidates(play, a, rng)
            expected, _, _ = _restricted_best(play, a, nb, cand, demands)
            res = br(play, a, current, cand)
            if current[a] == base.max_reward:
                # r_a = R_max: IP = 0 without a query (rcmas.ibis.at_reward_bound).
                assert expected == current[a]
                assert (res.status, res.reward, res.play, res.checks) == (
                    "ok",
                    current[a],
                    None,
                    0,
                )
                continue
            assert res.status == "ok" and res.play is not None
            assert res.reward == expected >= current[a]
            assert res.reward == rewards_from_owner(res.play.final_state, n, nb)[a]
            # Remark 4.9.1: the opponents follow the canonical completion.
            for step in range(res.play.horizon):
                s = res.play.states[step]
                for b in range(n):
                    if b != a:
                        assert res.play.actions[step][b] == comp.action(s, b)
            # The deviator keeps to its candidates at every candidate state.
            assert _respects(res.play, res.play.claims_of(a), cand)
            assert all(x >= y for x, y in zip(res.solution.payoff_by_agent, d))

    def test_current_action_only_returns_the_play_value(self):
        """With k = 0 the deviator cannot leave the play: IP = 0, never nil."""
        t = Territory.from_ascii(["...", "...", "..."])
        base = build_base_encoding(territory=t, num_agents=2, horizon=4)
        play = random_play(9, 2, 4, random.Random(7))
        current = rewards_from_owner(play.final_state, 2, t.neighbor_map())
        br = restricted_smt_best_response(base)
        for a, cand in enumerate(_current_only(play)):
            res = br(play, a, current, cand)
            assert res.status == "ok" and res.reward == current[a]
            assert res.play.claims_of(a) == play.claims_of(a)

    def test_unsatisfiable_restricted_query_is_an_error(self, monkeypatch):
        monkeypatch.setattr(smt_solve, "_check", lambda s, *a: z3.unsat)
        t = Territory.from_ascii(["..", ".."])
        base = build_base_encoding(territory=t, num_agents=2, horizon=2)
        play = Play.from_actions(4, 2, [[0, 1], [3, 2]])  # rewards (1, 1) < m = 2
        with pytest.raises(RuntimeError, match="unsatisfiable"):
            restricted_smt_best_response(base)(play, 0, (1, 1), _current_only(play)[0])


# ===================================================================
# The learners (lines 1, 11 and 13)
# ===================================================================


class TestLearner:
    def _learner(self, seed=0, k=3):
        t = Territory.from_ascii(["...", "...", "..."])
        game = LearningGame.from_territory(t, 2)
        return game, QibisLearner(game, LearningParams(), k=k, seed=seed)

    def test_candidate_sets(self):
        game, learner = self._learner()
        learner.pretrain(40)
        play = random_play(9, 2, 4, random.Random(1))
        cands = learner.refine(play, 40)
        assert len(cands) == 2
        for a, cand in enumerate(cands):
            for t in range(play.horizon):
                s = play.states[t]
                assert play.actions[t][a] in cand[s]
            for s, js in cand.items():
                assert is_live(s, 2)
                assert 1 <= len(js) <= 4  # TopK plus the current action
                assert all(s[j] == -1 for j in js)
            off_play = [s for s in cand if play.step_of(s) is None]
            assert off_play, "online training visits states off the play"
            assert all(len(cand[s]) <= 3 for s in off_play)

    def test_k_zero_keeps_only_current_actions(self):
        _, learner = self._learner(k=0)
        play = random_play(9, 2, 4, random.Random(2))
        cands = learner.refine(play, 20)
        assert cands[0] == {
            **{s: () for s in cands[0] if play.step_of(s) is None},
            **_current_only(play)[0],
        }

    def test_tables_persist_and_runs_are_reproducible(self):
        play = random_play(9, 2, 4, random.Random(3))
        _, one = self._learner(seed=5)
        _, two = self._learner(seed=5)
        one.pretrain(30)
        two.pretrain(30)
        after_pretrain = {s: dict(r) for s, r in one.tables[0].items()}
        assert one.refine(play, 30) == two.refine(play, 30)
        assert one.tables == two.tables
        assert set(after_pretrain) <= set(one.tables[0])
        assert one.tables[0] != after_pretrain

    def test_online_opponents_follow_completion(self, monkeypatch):
        """Line 11: while Q_a is trained, the other agents follow the completion."""
        import rcmas.qibis as qibis_mod

        seen = []
        real = qibis_mod.train_q

        def spy(q, game, agent, *, opponents, episodes, params, rng):
            seen.append(opponents)
            return real(
                q,
                game,
                agent,
                opponents=opponents,
                episodes=episodes,
                params=params,
                rng=rng,
            )

        monkeypatch.setattr(qibis_mod, "train_q", spy)
        _, learner = self._learner()
        learner.pretrain(2)
        assert seen == [None, None]
        play = random_play(9, 2, 4, random.Random(4))
        learner.refine(play, 2)
        comp = CanonicalCompletion(play)
        for opp in seen[2:]:
            s = play.states[1]
            assert opp(s, 0) == comp.action(s, 0) and opp(s, 1) == comp.action(s, 1)


# ===================================================================
# Q-IBIS on Z3: soundness against enumeration (Corollary 4.9.1)
# ===================================================================

SOUNDNESS_CASES = [
    (["..", ".."], 2, None, 3),
    (["...", "..."], 3, None, 3),
    (["...", "...", "..."], 2, None, 1),  # n does not divide |T|
    ([".....", "....."], 3, None, 3),  # n does not divide |T|
    (["...", "...", "..."], 3, (2, 2, 2), 1),
    (["..#", "...", ".#."], 2, (1, 1), 3),
]


class TestSoundness:
    @pytest.mark.parametrize("grid,n,demands,k", SOUNDNESS_CASES)
    def test_returned_equilibria_are_equilibria(self, grid, n, demands, k):
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        cfg = QibisConfig(top_k=k, demands=demands, **FAST)
        res = solve_qibis(territory=t, num_agents=n, cfg=cfg)
        assert res.outcome in ("ne", "n/c")
        assert res.play is not None and res.is_sat
        assert res.rewards == rewards_from_owner(res.play.final_state, n, nb)
        assert res.strategy == res.play.to_coords(t)
        assert res.final_solution is not None and res.final_solution.play == res.play
        d = demands or (0,) * n
        assert all(r >= x for r, x in zip(res.rewards, d))
        if res.outcome == "ne":
            assert is_equilibrium(res.play, nb, demands)
            last = res.trace[-1]
            assert last.restricted == (0,) * n and last.certified == (0,) * n
            comp = res.completion()
            assert comp is not None and comp.play == res.play

    def test_certification_from_figure_3_1(self):
        """k = 0 from the play of Figure 3.1: restricted (0, 0, 0), certified (1, 0, 0)."""
        t = Territory.from_ascii(RUNNING)
        cfg = QibisConfig(top_k=0, demands=(2, 2, 2), max_iters=0, **FAST)
        res = solve_qibis(
            territory=t, num_agents=3, cfg=cfg, initial_play=_fig_3_1_play()
        )
        it = res.trace[0]
        assert it.rewards == (2, 4, 4)
        assert it.restricted == (0, 0, 0) and it.certified == (1, 0, 0)
        assert it.chosen == 0 and res.rewards[0] == 3
        assert res.reason == "max_iters" and res.iterations == 0

    def test_certification_equals_ibis_potentials(self):
        """The certification round poses the IBIS query: equal potentials in iteration 0."""
        t = Territory.from_ascii(["...", "...", "..."])
        play = random_play(9, 2, 4, random.Random(11))
        cfg = QibisConfig(top_k=0, max_iters=0, **FAST)
        q = solve_qibis(territory=t, num_agents=2, cfg=cfg, initial_play=play)
        i = solve_ibis(territory=t, num_agents=2, max_iters=0, initial_play=play)
        expected = i.trace[0].improvement
        assert q.trace[0].improvement == expected
        if any(expected):
            assert q.trace[0].restricted == (0, 0)

    def test_weighted_abstract_game(self):
        """Q-IBIS runs on a weighted game with block adjacency (Section 6.1)."""
        abstract = Territory.from_ascii([".....", ""])
        w = (2, 1, 3, 1, 2)
        nb = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
        res = solve_qibis(
            territory=abstract,
            num_agents=2,
            cfg=QibisConfig(**FAST),
            weights=w,
            custom_neighbors=nb,
        )
        assert res.play is not None
        assert res.rewards == rewards_from_owner(res.play.final_state, 2, nb, w)
        if res.outcome == "ne":
            assert is_equilibrium(res.play, nb, None, w)


# ===================================================================
# Outcomes, solver answers and symmetry
# ===================================================================


class TestOutcomes:
    def test_no_satisfying_strategy(self):
        t = Territory.from_ascii(["...", "...", "..."])
        res = solve_qibis(
            territory=t, num_agents=3, cfg=QibisConfig(demands=(4, 0, 0), **FAST)
        )
        assert res.outcome == "uns" and res.reason == "unsat" and not res.is_sat
        assert res.play is None

    def test_unknown_in_initial_query(self, monkeypatch):
        monkeypatch.setattr(smt_solve, "_check", lambda solver, *a: z3.unknown)
        t = Territory.from_ascii(["..", ".."])
        res = solve_qibis(territory=t, num_agents=2, cfg=QibisConfig(**FAST))
        assert res.outcome == "unk" and res.reason == "unknown" and not res.is_sat

    @pytest.mark.parametrize("call,phase", [(1, "restricted"), (3, "certification")])
    def test_unknown_stops_the_run(self, monkeypatch, call, phase):
        """Checks: 1-2 restricted, 3-4 certification (k = 0, given play, no reward m).

        The initial play is given, so that no agent has reward m and every
        query is posed (an agent at m gets IP 0 without a query).
        """
        real = smt_solve._check
        calls = {"n": 0}

        def nth_unknown(s, *a):
            calls["n"] += 1
            return z3.unknown if calls["n"] == call else real(s, *a)

        monkeypatch.setattr(smt_solve, "_check", nth_unknown)
        t = Territory.from_ascii(["...", "..."])
        play = Play.from_actions(6, 2, [[0, 1], [2, 3], [4, 5]])
        assert play.rewards(t.neighbor_map()) == (1, 1)
        res = solve_qibis(
            territory=t,
            num_agents=2,
            cfg=QibisConfig(top_k=0, **FAST),
            initial_play=play,
        )
        assert res.outcome == "unk" and not res.found_ne and res.play is not None
        it = res.trace[0]
        if phase == "restricted":
            assert it.restricted == () and it.certified == ()
        else:
            assert it.restricted == (0, 0) and it.certified == ()

    def test_symmetry_only_in_initial_query(self, monkeypatch):
        calls = {"n": 0}
        real = smt_solve.symmetry_constraints

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(smt_solve, "symmetry_constraints", spy)
        t = Territory.from_ascii(["...", "...", "..."])
        res = solve_qibis(
            territory=t, num_agents=3, cfg=QibisConfig(symmetry=True, **FAST)
        )
        assert res.outcome in ("ne", "n/c") and calls["n"] == 1
        if res.outcome == "ne":
            assert is_equilibrium(res.play, t.neighbor_map())
        solve_qibis(territory=t, num_agents=3, cfg=QibisConfig(symmetry=False, **FAST))
        assert calls["n"] == 1


# ===================================================================
# Input validation and trivial games
# ===================================================================


class TestValidation:
    def test_zero_agents_raises(self):
        with pytest.raises(ValueError, match="num_agents"):
            solve_qibis(
                territory=Territory.from_ascii([".."]), num_agents=0, cfg=QibisConfig()
            )

    def test_horizon_must_be_m(self):
        with pytest.raises(ValueError, match="horizon"):
            solve_qibis(
                territory=Territory.from_ascii([".."]),
                num_agents=1,
                horizon=1,
                cfg=QibisConfig(),
            )

    def test_negative_parameters_raise(self):
        t = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="max_iters"):
            solve_qibis(territory=t, num_agents=1, cfg=QibisConfig(max_iters=-1))
        with pytest.raises(ValueError, match="top_k"):
            solve_qibis(territory=t, num_agents=1, cfg=QibisConfig(top_k=-1))

    def test_zero_iteration_bound_runs_one_iteration(self):
        t = Territory.from_ascii(["..", ".."])
        res = solve_qibis(
            territory=t, num_agents=2, cfg=QibisConfig(max_iters=0, **FAST)
        )
        assert len(res.trace) == 1 and res.iterations == 0

    def test_trivial_one_round(self):
        res = solve_qibis(
            territory=Territory.from_ascii([".."]),
            num_agents=2,
            horizon=1,
            cfg=QibisConfig(**FAST),
        )
        assert res.outcome == "ne" and res.rewards == (1, 1)

    def test_horizon_zero_game(self):
        """|T| < |Agt|: the empty play is an equilibrium (no agent acts)."""
        res = solve_qibis(
            territory=Territory.from_ascii(["."]), num_agents=2, cfg=QibisConfig(**FAST)
        )
        assert res.outcome == "ne" and res.rewards == (0, 0)

    def test_initial_play_must_match(self):
        t = Territory.from_ascii(["..", ".."])
        with pytest.raises(ValueError, match="initial_play"):
            solve_qibis(
                territory=t,
                num_agents=2,
                cfg=QibisConfig(**FAST),
                initial_play=Play.from_actions(4, 2, [[0, 1]]),
            )


# ===================================================================
# The protocol of Chapter 7: no iteration bound, collectively optimal start
# ===================================================================


class TestUnboundedAndCollective:
    def test_unbounded_loop_stops_at_recurrence(self):
        plays = _line_plays(2)
        back = {plays[0]: plays[1], plays[1]: plays[0]}
        res = run_qibis_loop(
            initial_play=plays[0],
            initial_solution=None,
            refine=lambda p: [{} for _ in range(3)],
            restricted_best_response=lambda p, a, cur, c: BestResponse(
                "ok", cur[a] + (a == 0), back[p] if a == 0 else None, None, 1
            ),
            best_response=None,
            rewards_of=lambda s: (1, 1, 1),
            max_iters=None,
        )
        assert res.outcome == "n/c" and res.reason == "cycle" and res.iterations == 2

    @pytest.mark.parametrize("initial", ["satisfying", "collective"])
    @pytest.mark.parametrize("symmetry", [False, True])
    @pytest.mark.parametrize(
        "grid,n,demands",
        [
            (["..", ".."], 2, None),
            (["...", "..."], 3, None),
            (["...", "...", "..."], 2, None),
            (["..#..", "..#.."], 2, None),
            (["...", "..."], 2, (2, 2)),
        ],
    )
    def test_unbounded_runs_terminate_and_are_sound(
        self, initial, symmetry, grid, n, demands
    ):
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        cfg = QibisConfig(
            max_iters=None, initial=initial, symmetry=symmetry, demands=demands, **FAST
        )
        res = solve_qibis(territory=t, num_agents=n, cfg=cfg)
        assert res.outcome in ("ne", "n/c") and res.reason in ("ne", "cycle")
        assert res.initial == initial and res.initial_query_s is not None
        assert res.initial_rewards == rewards_from_owner(
            res.initial_play.final_state, n, nb
        )
        if initial == "collective":
            m = len(t) // n
            assert sum(res.initial_rewards) == max_welfare(len(t), n, m, nb, demands)
        if res.outcome == "ne":
            assert is_equilibrium(res.play, nb, demands)
            if res.returned_initial:
                assert res.play == res.initial_play and res.iterations == 0

    def test_all_m_collective_play_returns_in_iteration_0(self):
        t = Territory.from_ascii(["...", "..."])
        cfg = QibisConfig(max_iters=None, initial="collective", **FAST)
        res = solve_qibis(territory=t, num_agents=2, cfg=cfg)
        assert res.initial_rewards == (3, 3)
        assert res.outcome == "ne" and res.iterations == 0 and res.returned_initial
        it = res.trace[0]
        assert it.restricted == (0, 0) and it.certified == (0, 0)  # certified as well
        assert is_equilibrium(res.play, t.neighbor_map())

    def test_unknown_collective_query_stops_with_unk(self, monkeypatch):
        monkeypatch.setattr(smt_solve, "_check", lambda solver, *a: z3.unknown)
        cfg = QibisConfig(initial="collective", **FAST)
        res = solve_qibis(
            territory=Territory.from_ascii(["..", ".."]), num_agents=2, cfg=cfg
        )
        assert res.outcome == "unk" and res.initial == "collective"
        assert res.initial_query_s is not None and res.initial_play is None

    def test_invalid_parameters_raise(self):
        t = Territory.from_ascii([".."])
        with pytest.raises(ValueError, match="initial"):
            solve_qibis(territory=t, num_agents=1, cfg=QibisConfig(initial="x"))
        with pytest.raises(ValueError, match="max_iters"):
            solve_qibis(territory=t, num_agents=1, cfg=QibisConfig(max_iters=-2))


# ===================================================================
# No query for an agent at R_max (rcmas.ibis.at_reward_bound)
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


class TestRewardBound:
    @pytest.mark.parametrize(
        "grid,n",
        [(["...", "..."], 2), (["...", "..."], 3), (["....", "...."], 4)],
    )
    @pytest.mark.parametrize("symmetry", [False, True])
    def test_all_m_start_poses_no_best_response_query(
        self, monkeypatch, grid, n, symmetry
    ):
        """Every reward m: neither restricted nor certification queries, only smt(F, f)."""
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        m = len(t) // n
        assert max_welfare(len(t), n, m, nb) == n * m
        checks = _count_checks(monkeypatch)
        cfg = QibisConfig(
            max_iters=None, initial="collective", symmetry=symmetry, **FAST
        )
        res = solve_qibis(territory=t, num_agents=n, cfg=cfg)
        assert res.initial_rewards == (m,) * n
        assert res.outcome == "ne" and res.iterations == 0 and res.returned_initial
        it = res.trace[0]
        assert it.restricted == (0,) * n and it.certified == (0,) * n
        assert it.checks == 0 and len(checks) == 1  # the initial query only
        assert is_equilibrium(res.play, nb)

    @pytest.mark.parametrize(
        "grid,n,demands,k,start",
        [
            (RUNNING, 3, (2, 2, 2), 0, "fig_3_1"),
            (RUNNING, 3, (2, 2, 2), 3, "fig_3_1"),
            (["....", "...."], 4, None, 3, "satisfying"),
            (["..#", "...", ".#."], 2, (1, 1), 1, "satisfying"),
            ([".....", "....."], 3, None, 3, "satisfying"),
        ],
    )
    def test_same_run_with_and_without_the_skip(
        self, monkeypatch, grid, n, demands, k, start
    ):
        """The skip removes queries only: same plays, potentials and outcome.

        The run without the skip poses every query, and the run with it is
        answered from that record (``tests/replay.py``).  The learners are
        seeded, so both runs train the same tables on the same plays.
        """
        t = Territory.from_ascii(grid)
        nb = t.neighbor_map()
        m = len(t) // n
        cfg = QibisConfig(top_k=k, demands=demands, max_iters=None, **FAST)
        kw: dict = dict(territory=t, num_agents=n, cfg=cfg)
        if start == "fig_3_1":
            kw["initial_play"] = _fig_3_1_play()
        replay = QueryReplay(monkeypatch)
        with monkeypatch.context() as mp:
            disable_skip(mp)
            full = solve_qibis(**kw)
        recorded = replay.replay()
        fast = solve_qibis(**kw)
        assert (fast.outcome, fast.reason, fast.iterations) == (
            full.outcome,
            full.reason,
            full.iterations,
        )
        assert fast.play == full.play and fast.rewards == full.rewards

        def rows(res):
            return [
                (it.index, it.rewards, it.restricted, it.certified, it.chosen)
                for it in res.trace
            ]

        assert rows(fast) == rows(full)
        # One query left out per round (restricted, certification) and agent at m.
        at_bound = 0
        for it in full.trace:
            for a in range(n):
                if it.rewards[a] == m:
                    assert a >= len(it.restricted) or it.restricted[a] == 0
                    assert a >= len(it.certified) or it.certified[a] == 0
                    at_bound += (a < len(it.restricted)) + (a < len(it.certified))
        assert at_bound > 0, "the case must exercise the skip"
        left_out = len(replay.best_responses(recorded)) - len(replay.best_responses())
        assert left_out == at_bound
        assert (
            sum(it.checks for it in full.trace) - sum(it.checks for it in fast.trace)
            == at_bound
        )
        if fast.outcome == "ne":
            assert is_equilibrium(fast.play, nb, demands)
