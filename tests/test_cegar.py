"""Tests for rcmas.cegar (thesis Algorithm 4, Sections 6.3-6.6).

Every equilibrium that CEGAR-NE returns is checked against an exhaustive
enumeration of the deviations from the canonical completion of its play
(``tests/brute_force.py``), on instances where ``n`` divides ``|T|`` and on
instances where it does not.  Adversarial inner synthesisers check that
soundness (Theorem 6.4.1) and the iteration bound (Theorem 6.4.2) do not
depend on the inner synthesiser.
"""

from __future__ import annotations

import random

import pytest

z3 = pytest.importorskip("z3")

from rcmas import smt_constraints, smt_solve  # noqa: E402
from rcmas.abstraction import (  # noqa: E402
    Partition,
    build_abstract_rcmas,
    discrete_partition,
    orbit_partition,
)
from rcmas.cegar import (  # noqa: E402
    CegarResult,
    InnerResult,
    SynthesisGame,
    make_synthesiser,
    solve_cegar,
    synthesis_game,
    verify_ne,
)
from rcmas.model import Territory  # noqa: E402
from rcmas.smt_solve import build_base_encoding  # noqa: E402
from rcmas.strategy import Play, canonical_completion, default_action  # noqa: E402

from .brute_force import best_deviation, is_equilibrium, random_play  # noqa: E402

GRID_4x2 = ["....", "...."]
GRID_6x2 = ["......", "......"]
GRID_3x3 = ["...", "...", "..."]


def _t(rows: list[str]) -> Territory:
    return Territory.from_ascii(rows)


def _check_run(
    res: CegarResult,
    territory: Territory,
    n: int,
    demands: tuple[int, ...] | None,
) -> None:
    """Theorems 6.4.1 and 6.4.2 on one run, with brute-force certification."""
    S = len(territory)
    bound = S - res.initial_partition_size + 1
    assert 1 <= res.iterations <= bound
    assert res.refinements == res.iterations - 1
    assert res.refinements <= S - res.initial_partition_size
    assert len(res.trace) == res.iterations
    for row in res.trace[:-1]:
        assert row.blocks_after > row.blocks
        assert row.refinement in ("bisect", "refine", "fallback")
    for a, b in zip(res.trace, res.trace[1:]):
        assert b.blocks == a.blocks_after
    assert res.final_partition_size == res.trace[-1].blocks
    if res.found_ne:
        assert res.outcome == "ne" and res.reason == "ne" and res.is_sat
        assert res.trace[-1].check == "verified"
        assert res.at_discrete == (res.final_partition_size == S)
        play = res.play
        assert play is not None and play.horizon == S // n
        d = demands or (0,) * n
        rewards = play.rewards(territory.neighbor_map())
        assert res.payoff_by_agent == rewards
        assert all(rewards[a] >= d[a] for a in range(n))
        assert is_equilibrium(play, territory.neighbor_map(), demands)
        comp = res.completion()
        assert comp is not None and comp.play == play
        assert res.final_solution is not None
        assert res.final_solution.owner_by_round == play.states
        assert res.strategy == play.to_coords(territory)
    else:
        assert res.completion() is None
        assert res.outcome in ("fail", "unk", "t/o", "n/c")


def _sound_inner(res: CegarResult) -> None:
    """With a sound inner synthesiser, a candidate at P_bot always passes VerifyNE.

    IBIS, S-IBIS and Q-IBIS return only plays that pass the unrestricted
    best-response test, so a run can end with FAILURE only through line 9
    (the inner synthesiser stops without an equilibrium of M), never through
    line 25.  Whether the inner synthesiser stops without an equilibrium
    depends on the models Z3 returns, so the outcome itself is not fixed.
    """
    assert res.outcome in ("ne", "fail")
    if res.outcome == "fail":
        assert res.reason == "failure_no_candidate"
    S = res.partition.num_sectors
    assert all(row.check != "refuted" for row in res.trace if row.blocks == S)


# ---------------------------------------------------------------------------
# VerifyNE (line 18)
# ---------------------------------------------------------------------------


class TestVerifyNE:
    @pytest.mark.parametrize(
        "rows, n, demands",
        [
            (["..", ".."], 2, None),
            (["...", "..."], 2, None),
            (["...", "..."], 2, (2, 1)),
            (GRID_3x3, 2, None),
            ([".....", "....."], 3, None),
            (["..#..", "..#.."], 2, (1, 2)),
        ],
    )
    def test_matches_exhaustive_deviations(self, rows, n, demands) -> None:
        """VerifyNE verifies a satisfying play iff no admissible deviation is profitable."""
        t = _t(rows)
        nbrs = t.neighbor_map()
        m = len(t) // n
        base = build_base_encoding(
            territory=t, num_agents=n, horizon=m, demands=demands
        )
        d = demands or (0,) * n
        rng = random.Random(len(t) * 10 + n)
        seen = {"verified": 0, "counterexample": 0}
        tried = 0
        while tried < 8:
            play = random_play(len(t), n, m, rng)
            r = play.rewards(nbrs)
            if any(r[a] < d[a] for a in range(n)):
                continue
            tried += 1
            vr = verify_ne(base, play)
            assert vr.status in ("verified", "counterexample")
            seen[vr.status] += 1
            assert (vr.status == "verified") == is_equilibrium(play, nbrs, demands)
            if vr.status == "verified":
                assert vr.queries == n
                continue
            a = vr.agent
            assert vr.queries == a + 1
            # The agents before a cannot improve; a can, and its deviation is
            # an admissible run against the canonical completion.
            completion = canonical_completion(play)
            for b in range(a):
                assert best_deviation(completion, b, nbrs, demands)[0] <= r[b]
            dev = vr.deviation
            r_dev = dev.rewards(nbrs)
            assert r_dev[a] > r[a]
            assert all(r_dev[b] >= d[b] for b in range(n))
            for t_ in range(m):
                for b in range(n):
                    if b != a:
                        assert dev.actions[t_][b] == completion.action(
                            dev.states[t_], b
                        )

    def test_known_equilibrium(self, territory_2x1: Territory) -> None:
        base = build_base_encoding(territory=territory_2x1, num_agents=2, horizon=1)
        vr = verify_ne(base, Play.from_actions(2, 2, [[0, 1]]))
        assert vr.status == "verified" and vr.agent is None and vr.queries == 2

    def test_unknown_is_not_unsat(self, monkeypatch, territory_2x2: Territory) -> None:
        base = build_base_encoding(territory=territory_2x2, num_agents=2, horizon=2)
        play = Play.from_actions(4, 2, [[0, 1], [2, 3]])
        monkeypatch.setattr(smt_solve, "_check", lambda solver, *a: z3.unknown)
        vr = verify_ne(base, play)
        assert vr.status == "unknown" and vr.agent == 0 and vr.queries == 1

    def test_wrong_game_rejected(self, territory_2x2: Territory) -> None:
        base = build_base_encoding(territory=territory_2x2, num_agents=2, horizon=2)
        with pytest.raises(ValueError):
            verify_ne(base, Play.from_actions(4, 2, [[0, 1]]))


# ---------------------------------------------------------------------------
# Algorithm 4 with the real synthesisers
# ---------------------------------------------------------------------------


SOUNDNESS_CASES = [
    (["..", ".."], 2, None),
    (["...", "..."], 2, None),
    (GRID_4x2, 2, (1, 1)),
    (GRID_3x3, 2, None),  # n does not divide |T|
    (GRID_3x3, 3, (2, 2, 3)),
    (["..", ".", ".."], 2, None),  # L shape, n does not divide |T|
    (["..#..", "..#.."], 2, (2, 2)),  # two components
    (["..", "..", "..", "..", ".."], 3, None),  # 5x2, n does not divide |T|
]


class TestSoundness:
    @pytest.mark.parametrize("rows, n, demands", SOUNDNESS_CASES)
    @pytest.mark.parametrize("partition", ["orbit", "trivial"])
    @pytest.mark.parametrize("symmetry", [False, True])
    def test_sibis_equilibria_are_equilibria(
        self, rows, n, demands, partition, symmetry
    ):
        t = _t(rows)
        res = solve_cegar(
            territory=t,
            num_agents=n,
            initial_partition=partition,
            synthesiser="sibis",
            symmetry=symmetry,
            demands=demands,
        )
        _check_run(res, t, n, demands)
        _sound_inner(res)

    @pytest.mark.parametrize("rows, n, demands", SOUNDNESS_CASES[:5])
    def test_ibis_equilibria_are_equilibria(self, rows, n, demands):
        t = _t(rows)
        res = solve_cegar(
            territory=t, num_agents=n, synthesiser="ibis", demands=demands
        )
        _check_run(res, t, n, demands)
        _sound_inner(res)

    @pytest.mark.parametrize(
        "rows, n, demands", [(GRID_4x2, 2, (1, 1)), (["...", "..."], 3, None)]
    )
    def test_qibis_equilibria_are_equilibria(self, rows, n, demands):
        t = _t(rows)
        res = solve_cegar(
            territory=t,
            num_agents=n,
            synthesiser="qibis",
            demands=demands,
            symmetry=True,
        )
        _check_run(res, t, n, demands)
        _sound_inner(res)

    def test_non_dividing_3x3_is_not_accepted_at_rewards_1_1(self) -> None:
        """Defect 7: with require_victory the old loop accepted (1, 1) on 3x3 with n = 2."""
        t = _t(GRID_3x3)
        # Agent 1 holds the corners, agent 2 the edges: each agent can reach 4.
        corners_edges = Play.from_actions(9, 2, [[0, 1], [2, 3], [6, 5], [8, 7]])
        assert corners_edges.rewards(t.neighbor_map()) == (1, 1)
        base = build_base_encoding(territory=t, num_agents=2, horizon=4)
        vr = verify_ne(base, corners_edges)
        assert vr.status == "counterexample" and vr.agent == 0
        res = solve_cegar(territory=t, num_agents=2, synthesiser="sibis")
        _check_run(res, t, 2, None)
        _sound_inner(res)
        if res.found_ne:
            assert res.payoff_by_agent != (1, 1)


class TestExamples:
    def test_example_6_3_2_iteration_1(self) -> None:
        """Example 6.3.2: the half partition of the 4x2 grid is verified at once."""
        t = _t(GRID_4x2)
        halves = Partition.from_blocks([{0, 1, 4, 5}, {2, 3, 6, 7}], 8)
        res = solve_cegar(
            territory=t, num_agents=2, initial_partition=halves, demands=(1, 1)
        )
        _check_run(res, t, 2, (1, 1))
        assert res.outcome == "ne" and res.iterations == 1 and res.refinements == 0
        assert res.final_partition_size == 2 and not res.at_discrete
        assert res.payoff_by_agent == (4, 4)
        claims = {res.play.claims_of(0), res.play.claims_of(1)}
        assert claims == {(0, 1, 4, 5), (2, 3, 6, 7)}
        assert res.trace[0].abstract_rewards == (4, 4)
        assert res.trace[0].verify_queries == 2

    def test_example_6_3_2_alternative_scenario(self) -> None:
        """Example 6.3.2: m_hat = 0, no satisfying abstract strategy, bisect {1,2,3,7,8,9}."""
        t = _t(GRID_6x2)
        P0 = Partition.from_blocks([{0, 1, 2, 6, 7, 8}, {3, 4, 5, 9, 10, 11}], 12)
        seen: list[Partition] = []
        sibis = make_synthesiser("sibis")

        def recording(game: SynthesisGame) -> InnerResult:
            seen.append(game.abstract.partition)
            return sibis(game)

        # The example follows the first iteration only; the cap keeps the test
        # short (later iterations depend on the models Z3 returns).
        res = solve_cegar(
            territory=t,
            num_agents=3,
            initial_partition=P0,
            synthesiser=recording,
            demands=(1, 1, 1),
            max_refinements=1,
        )
        _check_run(res, t, 3, (1, 1, 1))
        assert res.iterations <= 2
        first = res.trace[0]
        assert first.synthesis == "uns" and first.check == "no_candidate"
        assert first.refinement == "bisect" and first.blocks_after == 3
        assert seen[1].blocks == (
            frozenset({0, 1, 6}),
            frozenset({2, 7, 8}),
            frozenset({3, 4, 5, 9, 10, 11}),
        )


# ---------------------------------------------------------------------------
# Algorithm 4 with adversarial inner synthesisers (Theorems 6.4.1 and 6.4.2)
# ---------------------------------------------------------------------------


def _never(game: SynthesisGame) -> InnerResult:
    return InnerResult("n/c")


def _default_play(game: SynthesisGame) -> InnerResult:
    """The play in which every agent always takes its default action."""
    n, k, h = game.num_agents, len(game.territory), game.horizon
    state = [-1] * k
    actions = []
    for _ in range(h):
        prof = [default_action(tuple(state), a, n) for a in range(n)]
        for a, i in enumerate(prof):
            state[i] = a
        actions.append(prof)
    return InnerResult("ne", Play.from_actions(k, n, actions))


def _random_synth(seed: int):
    rng = random.Random(seed)

    def run(game: SynthesisGame) -> InnerResult:
        if rng.random() < 0.2:
            return InnerResult("uns")
        k = len(game.territory)
        return InnerResult("ne", random_play(k, game.num_agents, game.horizon, rng))

    return run


class TestAdversarialSynthesisers:
    @pytest.mark.parametrize(
        "rows, n, partition",
        [(GRID_3x3, 2, "trivial"), (GRID_3x3, 3, "orbit"), (GRID_6x2, 3, "orbit")],
    )
    def test_no_candidate_reaches_the_bound(self, rows, n, partition) -> None:
        """With no candidate ever, each iteration splits one block (Theorem 6.4.2 is tight)."""
        t = _t(rows)
        res = solve_cegar(
            territory=t, num_agents=n, initial_partition=partition, synthesiser=_never
        )
        _check_run(res, t, n, None)
        assert res.outcome == "fail" and res.reason == "failure_no_candidate"
        assert res.iterations == len(t) - res.initial_partition_size + 1
        assert res.final_partition_size == len(t)
        assert all(r.blocks_after == r.blocks + 1 for r in res.trace[:-1])
        assert res.play is None

    def test_refuted_at_discrete_partition_is_failure(self) -> None:
        """A candidate refuted at P_bot ends the run with FAILURE (line 25)."""
        t = _t(GRID_3x3)
        res = solve_cegar(
            territory=t,
            num_agents=2,
            initial_partition="discrete",
            synthesiser=_default_play,
        )
        _check_run(res, t, 2, None)
        assert res.outcome == "fail" and res.reason == "failure_refuted"
        assert res.iterations == 1
        assert res.trace[0].check == "refuted"
        assert res.payoff_by_agent == (1, 1)
        assert not is_equilibrium(res.play, t.neighbor_map())

    def test_refuted_until_discrete(self) -> None:
        """Default plays from {T}: a lifted default play can be an NE at a coarse
        partition (which _check_run certifies); otherwise the run is refuted at P_bot.
        """
        t = _t(GRID_3x3)
        res = solve_cegar(
            territory=t,
            num_agents=2,
            initial_partition="trivial",
            synthesiser=_default_play,
        )
        _check_run(res, t, 2, None)
        assert res.outcome in ("ne", "fail")
        if res.outcome == "fail":
            assert res.reason == "failure_refuted"
            assert res.trace[-1].blocks == 9 and res.trace[-1].check == "refuted"

    def test_unsatisfying_candidate_at_discrete_partition(self) -> None:
        t = _t(GRID_3x3)
        res = solve_cegar(
            territory=t,
            num_agents=2,
            initial_partition="discrete",
            synthesiser=_default_play,
            demands=(2, 2),
        )
        assert res.outcome == "fail" and res.reason == "failure_refuted"
        assert res.trace[0].check == "unsatisfying" and res.trace[0].agent == 0
        assert res.trace[0].verify_queries == 0

    @pytest.mark.parametrize("seed", range(6))
    def test_random_candidates(self, seed: int) -> None:
        """Soundness and termination with an unsound, random inner synthesiser."""
        rows, n, demands = [
            (GRID_3x3, 2, None),
            (["...", "..."], 2, (1, 1)),
            (GRID_4x2, 2, None),
            (["..", ".", ".."], 2, None),
            (["...", "..."], 3, None),
            (GRID_3x3, 3, (1, 1, 1)),
        ][seed]
        t = _t(rows)
        for partition in ("trivial", "orbit"):
            res = solve_cegar(
                territory=t,
                num_agents=n,
                initial_partition=partition,
                synthesiser=_random_synth(seed),
                demands=demands,
            )
            _check_run(res, t, n, demands)
            assert res.outcome in ("ne", "fail")

    def test_fallback_bisects_a_block_inside_D(self) -> None:
        """D a union of blocks: bisect the largest block of D with more than one sector."""
        t = _t(GRID_4x2)
        columns = Partition.from_blocks([{0, 4}, {1, 5}, {2, 6}, {3, 7}], 8)
        seen: list[Partition] = []

        def split_columns(game: SynthesisGame) -> InnerResult:
            seen.append(game.abstract.partition)
            if len(seen) == 1:
                # Agent 0 takes the non-adjacent columns 1 and 3 (reward 2 < 4).
                return InnerResult("ne", Play.from_actions(4, 2, [[0, 1], [2, 3]]))
            return InnerResult("uns")

        res = solve_cegar(
            territory=t,
            num_agents=2,
            initial_partition=columns,
            synthesiser=split_columns,
            demands=(4, 0),
        )
        first = res.trace[0]
        assert first.check == "unsatisfying" and first.agent == 0
        assert first.deviation_size == 4 and first.refinement == "fallback"
        assert seen[1].blocks == (
            frozenset({0}),
            frozenset({1, 5}),
            frozenset({2, 6}),
            frozenset({3, 7}),
            frozenset({4}),
        )
        _check_run(res, t, 2, (4, 0))

    def test_fallback_never_picks_a_singleton_in_D(self) -> None:
        """D a union of singletons: bisect the largest block of P instead."""
        t = _t(["..", ".."])  # sectors 0, 1 bottom row, 2, 3 top row
        P0 = Partition.from_blocks([{0}, {1, 3}, {2}], 4)
        seen: list[Partition] = []

        def synth(game: SynthesisGame) -> InnerResult:
            seen.append(game.abstract.partition)
            if len(seen) == 1:
                # Agent 0 claims {0}, agent 1 claims {1, 3}; sector 2 goes to agent 0.
                return InnerResult("ne", Play.from_actions(3, 2, [[0, 1]]))
            return InnerResult("uns")

        res = solve_cegar(
            territory=t,
            num_agents=2,
            initial_partition=P0,
            synthesiser=synth,
            demands=(3, 0),
        )
        first = res.trace[0]
        assert first.check == "unsatisfying" and first.refinement == "fallback"
        assert res.play is not None
        assert seen[1].is_discrete
        _check_run(res, t, 2, (3, 0))


# ---------------------------------------------------------------------------
# Symmetry (Section 6.5), unknown answers, limits and parameters
# ---------------------------------------------------------------------------


class TestSymmetryScope:
    @pytest.mark.parametrize("symmetry", [False, True])
    def test_symmetry_only_in_initial_queries(self, monkeypatch, symmetry) -> None:
        calls = {"abstract": 0, "grid": 0}
        real_abstract = smt_constraints.symmetry_breaking_constraint
        real_grid = smt_solve.symmetry_constraints

        def abstract_spy(*a, **kw):
            calls["abstract"] += 1
            return real_abstract(*a, **kw)

        def grid_spy(*a, **kw):
            calls["grid"] += 1
            return real_grid(*a, **kw)

        monkeypatch.setattr(
            smt_constraints, "symmetry_breaking_constraint", abstract_spy
        )
        monkeypatch.setattr(smt_solve, "symmetry_constraints", grid_spy)
        t = _t(GRID_3x3)
        res = solve_cegar(territory=t, num_agents=3, symmetry=symmetry)
        _check_run(res, t, 3, None)
        abstract_iters = sum(1 for r in res.trace if r.blocks < len(t))
        discrete_iters = res.iterations - abstract_iters
        if symmetry:
            # One initial query per synthesis call, nothing in BR or VerifyNE.
            assert calls["abstract"] == abstract_iters
            assert calls["grid"] == discrete_iters
        else:
            assert calls == {"abstract": 0, "grid": 0}

    def test_abstract_game_takes_no_grid_symmetry(self) -> None:
        t = _t(["...."] * 4)
        A = build_abstract_rcmas(t, 2, orbit_partition(t))
        g = synthesis_game(A, symmetry=True)
        assert not g.symmetry and g.initial_constraints is not None
        assert g.weights == (4, 8, 4) and g.custom_neighbors == {
            0: [1],
            1: [0, 2],
            2: [1],
        }
        assert synthesis_game(A, symmetry=False).initial_constraints is None
        D = build_abstract_rcmas(t, 2, discrete_partition(t))
        gd = synthesis_game(D, symmetry=True)
        assert gd.is_concrete and gd.symmetry and gd.territory is t and gd.horizon == 8


class TestUnknown:
    def test_unknown_in_inner_synthesis(self, monkeypatch, territory_2x2) -> None:
        monkeypatch.setattr(smt_solve, "_check", lambda solver, *a: z3.unknown)
        res = solve_cegar(territory=territory_2x2, num_agents=2)
        assert res.outcome == "unk" and res.reason == "unknown"
        assert res.iterations == 1 and res.trace[0].check == "unknown"
        assert not res.found_ne

    def test_unknown_in_verification(self, monkeypatch) -> None:
        t = _t(GRID_3x3)
        monkeypatch.setattr(smt_solve, "_check", lambda solver, *a: z3.unknown)
        res = solve_cegar(
            territory=t,
            num_agents=2,
            initial_partition="trivial",
            synthesiser=_default_play,
        )
        assert res.outcome == "unk" and res.iterations == 1
        assert res.trace[0].check == "unknown" and res.trace[0].verify_queries == 1

    def test_real_timeout_answers_unknown(self) -> None:
        t = _t(["......"] * 4)
        res = solve_cegar(territory=t, num_agents=2, timeout_ms=1)
        assert res.outcome in ("unk", "ne")
        if res.outcome == "ne":
            _check_run(res, t, 2, None)


class TestLimitsAndParameters:
    def test_time_limit(self, territory_3x3) -> None:
        res = solve_cegar(territory=territory_3x3, num_agents=2, time_limit_s=0.0)
        assert res.outcome == "t/o" and res.iterations == 0 and not res.found_ne

    def test_max_refinements(self, territory_3x3) -> None:
        res = solve_cegar(
            territory=territory_3x3,
            num_agents=2,
            initial_partition="trivial",
            synthesiser=_never,
            max_refinements=2,
        )
        assert res.outcome == "n/c" and res.reason == "max_refinements"
        assert res.refinements == 2 and res.iterations == 3

    def test_inner_iteration_bound_matches_direct_runs(self, monkeypatch) -> None:
        import rcmas.ibis as ibis_mod

        bounds: list[int] = []
        real = ibis_mod.solve_sibis

        def spy(**kw):
            bounds.append(kw["max_iters"])
            return real(**kw)

        monkeypatch.setattr(ibis_mod, "solve_sibis", spy)
        t = _t(["...", "..."])
        solve_cegar(territory=t, num_agents=2)
        solve_cegar(territory=t, num_agents=2, max_iters=7)
        assert bounds and set(bounds) <= {30, 7} and bounds[0] == 30 and bounds[-1] == 7

    def test_inner_bound_zero(self) -> None:
        """N = 0: the inner synthesiser returns only an initial play that is already an NE."""
        t = _t(GRID_3x3)
        res = solve_cegar(territory=t, num_agents=3, max_iters=0)
        _check_run(res, t, 3, None)

    def test_parameter_validation(self, territory_2x2) -> None:
        with pytest.raises(ValueError, match="horizon"):
            solve_cegar(territory=territory_2x2, num_agents=2, horizon=3)
        with pytest.raises(ValueError, match="'ibis', 'qibis', or 'sibis'"):
            solve_cegar(territory=territory_2x2, num_agents=2, synthesiser="bogus")
        with pytest.raises(ValueError, match="unknown partition"):
            solve_cegar(
                territory=territory_2x2, num_agents=2, initial_partition="bogus"
            )
        with pytest.raises(ValueError):
            solve_cegar(territory=territory_2x2, num_agents=2, demands=(1,))
        with pytest.raises(ValueError):
            solve_cegar(territory=territory_2x2, num_agents=0)
        with pytest.raises(ValueError):
            solve_cegar(territory=territory_2x2, num_agents=2, max_iters=-1)

    def test_horizon_none_selects_m(self, territory_2x1) -> None:
        res = solve_cegar(territory=territory_2x1, num_agents=2, horizon=None)
        assert res.found_ne and res.play.horizon == 1
        res = solve_cegar(territory=territory_2x1, num_agents=2, horizon=1)
        assert res.found_ne

    def test_discrete_partition_is_verified(self, territory_2x2) -> None:
        """At P_bot the inner result is verified, not returned unchecked."""
        res = solve_cegar(
            territory=territory_2x2, num_agents=2, initial_partition="discrete"
        )
        _check_run(res, territory_2x2, 2, None)
        assert res.outcome == "ne" and res.at_discrete and res.reason == "ne"
        assert res.trace[0].verify_queries == 2
