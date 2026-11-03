"""Tests for smt_constraints.py (thesis Sections 4.2, 4.3 and 4.7).

The oracle tests compare the encoding with brute-force enumeration:

- the models of [Init] and [Evol] and [Prot] and [Dom] are exactly the plays
  (Lemma 4.2.1);
- on every final state of 2x2, 2x3 and 3x3 with 2 and 3 agents, the witness
  encoding admits ``reward_a >= c`` exactly when ``c <= r_a`` (Definition
  3.9.2), so its maximum is the reward and no model exceeds it
  (Lemmas 4.3.1, 4.3.2, Corollary 4.3.3), also with block weights;
- the models of the dynamics with [Fix^a_pi] are exactly the deviations of
  ``a`` against the canonical completion of ``pi`` (Lemma 4.7.1).
"""

from __future__ import annotations

import random

import pytest

from rcmas.model import Territory, rewards_from_owner
from rcmas.smt_constraints import (
    action_candidates_constraint,
    domain_constraint,
    evolution_constraint,
    fixed_policy_constraint,
    init_constraint,
    opponent_constraint,
    protocol_constraint,
    qualitative_constraint,
    reward_bound_constraint,
    reward_constraint,
    witness_constraint,
)
from rcmas.smt_variables import create_variables
from rcmas.strategy import CanonicalCompletion, Play

from .brute_force import deviations, final_states, random_play

z3 = pytest.importorskip("z3")


def _dynamics(solver, v) -> None:
    init_constraint(solver, v)
    evolution_constraint(solver, v)
    protocol_constraint(solver, v)
    domain_constraint(solver, v)


def _action_models(solver, v) -> set[tuple[tuple[int, ...], ...]]:
    """All values of the action variables over the models of *solver*."""
    out = set()
    while solver.check() == z3.sat:
        mdl = solver.model()
        acts = tuple(
            tuple(
                mdl.eval(v.action[a][t], model_completion=True).as_long()
                for a in range(v.A)
            )
            for t in range(v.T)
        )
        out.add(acts)
        solver.add(
            z3.Or(
                [v.action[a][t] != acts[t][a] for t in range(v.T) for a in range(v.A)]
            )
        )
    return out


def _all_plays(S: int, n: int, m: int) -> set[tuple[tuple[int, ...], ...]]:
    out = set()

    def rec(t, free, acts):
        if t == m:
            out.add(tuple(acts))
            return
        import itertools

        for prof in itertools.permutations(sorted(free), n):
            rec(t + 1, free - set(prof), acts + [prof])

    rec(0, set(range(S)), [])
    return out


# ===================================================================
# Dynamics: [Init], [Evol], [Prot], [Dom] (Definitions 4.2.1-4.2.4)
# ===================================================================


class TestDynamics:
    def test_init_all_unoccupied(self):
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 1)
        s = z3.Solver()
        init_constraint(s, v)
        s.add(v.owner[0][0] != -1)
        assert s.check() == z3.unsat

    @pytest.mark.parametrize(
        "grid,n",
        [(["..", ".."], 2), (["...", "..."], 2), (["...", "..."], 3), (["..."], 2)],
    )
    def test_models_are_exactly_the_plays(self, grid, n):
        """Lemma 4.2.1: the action variables of the models are the plays of M."""
        t = Territory.from_ascii(grid)
        m = len(t) // n
        v = create_variables(t, n, m)
        s = z3.Solver()
        _dynamics(s, v)
        assert _action_models(s, v) == _all_plays(len(t), n, m)

    def test_collision_excluded(self):
        """Clause 4 of Definition 4.2.2."""
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 2)
        s = z3.Solver()
        _dynamics(s, v)
        s.add(v.action[0][0] == v.action[1][0])
        assert s.check() == z3.unsat

    def test_collision_excluded_three_agents(self):
        t = Territory.from_ascii(["...", "..."])
        v = create_variables(t, 3, 2)
        s = z3.Solver()
        _dynamics(s, v)
        s.add(v.action[0][1] == v.action[2][1])
        assert s.check() == z3.unsat

    def test_domain_excludes_skipped_round(self):
        """Without [Dom] an out-of-range action would act as a pass."""
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 2)
        s = z3.Solver()
        init_constraint(s, v)
        evolution_constraint(s, v)
        protocol_constraint(s, v)
        s.add(v.action[0][0] == len(t))
        assert s.check() == z3.sat
        domain_constraint(s, v)
        assert s.check() == z3.unsat

    def test_protocol_forbids_occupied_claim(self):
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 1, 2)
        s = z3.Solver()
        _dynamics(s, v)
        s.add(v.action[0][0] == 1, v.action[0][1] == 1)
        assert s.check() == z3.unsat


# ===================================================================
# Witness and reward encoding (Definitions 4.3.1, 4.3.2)
# ===================================================================


def _witness_solver(v):
    s = z3.Solver()
    witness_constraint(s, v)
    reward_constraint(s, v)
    return s


def _check_witness_on_states(
    t: Territory, n: int, m: int, states, weights=None, neighbors=None
) -> int:
    """Lower-bound exactness on every given final state; returns the number of states."""
    v = create_variables(t, n, m, weights=weights, custom_neighbors=neighbors)
    s = _witness_solver(v)
    nb = v.neighbors
    count = 0
    for state in states:
        r = rewards_from_owner(state, n, nb, weights)
        s.push()
        s.add(*[v.owner[i][m] == state[i] for i in range(v.S)])
        for a in range(n):
            assert s.check(v.reward[a] >= r[a]) == z3.sat, (state, a)
            assert s.check(v.reward[a] >= r[a] + 1) == z3.unsat, (state, a)
        assert s.check(*[v.reward[a] == r[a] for a in range(n)]) == z3.sat, state
        s.pop()
        count += 1
    return count


class TestWitnessOracle:
    @pytest.mark.parametrize(
        "grid,n",
        [
            (["..", ".."], 2),
            (["..", ".."], 3),
            (["...", "..."], 2),
            (["...", "..."], 3),
            (["...", "...", "..."], 2),
            (["...", "...", "..."], 3),
        ],
    )
    def test_all_final_states(self, grid, n):
        """reward_a >= c is satisfiable iff c <= r_a, on every final state."""
        t = Territory.from_ascii(grid)
        m = len(t) // n
        states = list(final_states(len(t), n, m))
        assert _check_witness_on_states(t, n, m, states) == len(states)

    def test_maximisation_equals_reward_sample(self):
        """Optimize on reward_a returns r_a (sample of 3x3 final states, 2 agents)."""
        t = Territory.from_ascii(["...", "...", "..."])
        n, m = 2, 4
        v = create_variables(t, n, m)
        nb = t.neighbor_map()
        for k, state in enumerate(final_states(9, n, m)):
            if k % 45:
                continue
            r = rewards_from_owner(state, n, nb)
            for a in range(n):
                opt = z3.Optimize()
                witness_constraint(opt, v)
                reward_constraint(opt, v)
                opt.add(*[v.owner[i][m] == state[i] for i in range(9)])
                opt.maximize(v.reward[a])
                assert opt.check() == z3.sat
                assert opt.model().eval(v.reward[a]).as_long() == r[a]

    def test_l_tromino_scores_three(self):
        """The old interval recurrence scored this L-tromino 2."""
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 1, 3)
        s = _witness_solver(v)
        s.add(*[v.owner[i][3] == o for i, o in enumerate((0, -1, 0, 0))])
        assert s.check(v.reward[0] >= 3) == z3.sat

    def test_root_is_least_index_member(self):
        t = Territory.from_ascii(["..."])
        v = create_variables(t, 1, 3)
        s = _witness_solver(v)
        s.add(*[v.owner[i][3] == 0 for i in range(3)])
        s.add(v.reward[0] == 3, v.root[0][1])
        assert s.check() == z3.unsat

    def test_horizon_zero_is_satisfiable(self):
        """Level bound max(m-1, 0): with m = 0 the empty witness is a model."""
        t = Territory.from_ascii(["."])
        v = create_variables(t, 2, 0)
        s = _witness_solver(v)
        _dynamics(s, v)
        qualitative_constraint(s, v, (0, 0))
        assert s.check() == z3.sat

    def test_weighted_abstract_games(self):
        """Random partitions of 3x3: weighted witness exact for lower bounds (Section 6.1)."""
        t = Territory.from_ascii(["...", "...", "..."])
        nb3 = t.neighbor_map()
        rng = random.Random(7)
        checked = 0
        for k in (4, 5, 6):
            for _ in range(3):
                labels = list(range(k)) + [rng.randrange(k) for _ in range(9 - k)]
                rng.shuffle(labels)
                blocks = [
                    sorted(i for i in range(9) if labels[i] == p) for p in range(k)
                ]
                blocks.sort(key=lambda b: b[0])
                member = {i: p for p, b in enumerate(blocks) for i in b}
                weights = tuple(len(b) for b in blocks)
                adj = {
                    p: sorted({member[j] for i in blocks[p] for j in nb3[i]} - {p})
                    for p in range(k)
                }
                abstract = Territory.from_ascii(["." * k])
                n = 2
                m = k // n
                states = list(final_states(k, n, m))
                checked += _check_witness_on_states(
                    abstract, n, m, states, weights, adj
                )
        assert checked > 0


class TestQualitativeConstraint:
    def test_infeasible_demand_unsat(self):
        t = Territory.from_ascii(["...", "...", "..."])
        v = create_variables(t, 3, 3)
        s = z3.Solver()
        _dynamics(s, v)
        witness_constraint(s, v)
        reward_constraint(s, v)
        qualitative_constraint(s, v, (2, 2, 4))
        assert s.check() == z3.unsat

    def test_feasible_demand_sat_and_met(self):
        t = Territory.from_ascii(["...", "...", "..."])
        v = create_variables(t, 3, 3)
        s = z3.Solver()
        _dynamics(s, v)
        witness_constraint(s, v)
        reward_constraint(s, v)
        qualitative_constraint(s, v, (3, 3, 3))
        assert s.check() == z3.sat
        mdl = s.model()
        final = tuple(mdl.eval(v.owner[i][3]).as_long() for i in range(9))
        assert rewards_from_owner(final, 3, t.neighbor_map()) == (3, 3, 3)

    def test_wrong_length_raises(self):
        t = Territory.from_ascii([".."])
        v = create_variables(t, 2, 1)
        with pytest.raises(ValueError):
            qualitative_constraint(z3.Solver(), v, (1,))


# ===================================================================
# Opponent constraint [Fix^a_pi] (Definition 4.7.1, Lemma 4.7.1)
# ===================================================================


class TestOpponentConstraint:
    @pytest.mark.parametrize(
        "grid,n,seed",
        [(["...", "..."], 2, 0), (["...", "..."], 3, 1), (["...", "...", "..."], 2, 2)],
    )
    def test_models_are_the_deviations(self, grid, n, seed):
        """Lemma 4.7.1: models of dynamics and [Fix] = runs against the completion."""
        t = Territory.from_ascii(grid)
        m = len(t) // n
        play = random_play(len(t), n, m, random.Random(seed))
        comp = CanonicalCompletion(play)
        for a in range(n):
            v = create_variables(t, n, m)
            s = z3.Solver()
            _dynamics(s, v)
            opponent_constraint(s, v, play, a)
            expected = {
                run.actions for _, run in deviations(comp.action, a, len(t), n, m)
            }
            assert _action_models(s, v) == expected

    def test_return_to_play_resumes_recorded_actions(self):
        """A run that leaves the play and returns to it follows the record again."""
        # 1x6 line, 2 agents; play: a claims 0, 1, 5 and b claims 2, 3, 4.
        t = Territory.from_ascii(["......"])
        n = 2
        play = Play.from_actions(6, n, [[0, 2], [1, 3], [5, 4]])
        v = create_variables(t, n, 3)
        s = z3.Solver()
        _dynamics(s, v)
        opponent_constraint(s, v, play, 0)
        # a claims 1 then 0: the run leaves the play at step 1 (b takes its
        # default, the second unoccupied sector 3) and is back on s_2.
        s.add(v.action[0][0] == 1, v.action[0][1] == 0)
        assert s.check() == z3.sat
        mdl = s.model()
        assert mdl.eval(v.action[1][0]).as_long() == 2
        assert mdl.eval(v.action[1][1]).as_long() == 3
        # At s_2 b follows its record (4); its default would be 5.
        assert mdl.eval(v.action[1][2]).as_long() == 4

    def test_fixed_policy_with_completion_policy_is_equivalent(self):
        t = Territory.from_ascii(["...", "..."])
        n, m = 2, 3
        play = random_play(6, n, m, random.Random(3))
        comp = CanonicalCompletion(play)
        v = create_variables(t, n, m)
        s1 = z3.Solver()
        _dynamics(s1, v)
        opponent_constraint(s1, v, play, 0)
        s2 = z3.Solver()
        _dynamics(s2, v)
        fixed_policy_constraint(s2, v, t, (None, comp.policy(t, 1)))
        assert _action_models(s1, v) == _action_models(s2, v)

    def test_mismatched_play_raises(self):
        t = Territory.from_ascii(["...", "..."])
        v = create_variables(t, 2, 3)
        play = Play.from_actions(6, 2, [[0, 1]])
        with pytest.raises(ValueError):
            opponent_constraint(z3.Solver(), v, play, 0)


# ===================================================================
# Structural constraints
# ===================================================================


class TestFixedPolicyConstraint:
    def test_wrong_state_key_length_raises(self):
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 2)
        with pytest.raises(ValueError):
            fixed_policy_constraint(
                z3.Solver(), v, t, ({(-1,): t.ordered_sectors()[0]}, None)
            )

    def test_unmapped_states_take_default(self):
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 2)
        s = z3.Solver()
        _dynamics(s, v)
        fixed_policy_constraint(s, v, t, ({}, None))
        assert s.check() == z3.sat
        mdl = s.model()
        assert mdl.eval(v.action[0][0]).as_long() == 0


class TestActionCandidatesConstraint:
    def test_restricts_to_candidates(self):
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 2)
        sectors = t.ordered_sectors()
        s = z3.Solver()
        _dynamics(s, v)
        action_candidates_constraint(s, v, t, ({(-1, -1, -1, -1): (sectors[3],)}, None))
        assert s.check() == z3.sat
        assert s.model().eval(v.action[0][0]).as_long() == 3
        s.add(v.action[0][0] != 3)
        assert s.check() == z3.unsat

    def test_wrong_length_raises(self):
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 2)
        with pytest.raises(ValueError):
            action_candidates_constraint(z3.Solver(), v, t, (None,))


class TestRewardBound:
    """``reward_a <= R_max`` is implied by F and removes no model (Section 7.1)."""

    @pytest.mark.parametrize(
        "grid,n", [(["...", "..."], 2), (["...", "...", "..."], 3)]
    )
    def test_witness_oracle_unchanged(self, grid, n):
        t = Territory.from_ascii(grid)
        m = len(t) // n
        v = create_variables(t, n, m)
        s = z3.Solver()
        witness_constraint(s, v)
        reward_constraint(s, v)
        reward_bound_constraint(s, v)
        nb = t.neighbor_map()
        for state in final_states(len(t), n, m):
            r = rewards_from_owner(state, n, nb)
            s.push()
            s.add(*[v.owner[i][m] == state[i] for i in range(v.S)])
            assert s.check(*[v.reward[a] == r[a] for a in range(n)]) == z3.sat
            s.pop()

    def test_plays_unchanged(self):
        t = Territory.from_ascii(["...", "..."])
        v = create_variables(t, 2, 3)
        s = z3.Solver()
        _dynamics(s, v)
        witness_constraint(s, v)
        reward_constraint(s, v)
        reward_bound_constraint(s, v)
        assert _action_models(s, v) == _all_plays(6, 2, 3)

    def test_weighted_bound(self):
        t = Territory.from_ascii(["...."])
        v = create_variables(
            t,
            2,
            2,
            weights=(1, 4, 2, 3),
            custom_neighbors={0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
        )
        s = z3.Solver()
        reward_bound_constraint(s, v)
        assert s.check(v.reward[0] == 7) == z3.sat
        assert s.check(v.reward[0] == 8) == z3.unsat
