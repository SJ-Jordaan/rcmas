"""Tests for strategy.py (thesis Sections 3.8 and 4.4)."""

from __future__ import annotations

import pytest

from rcmas.model import CollisionError, Coord, State, Territory, evolve
from rcmas.strategy import (
    CanonicalCompletion,
    Play,
    Policy,
    canonical_completion,
    default_action,
    fallback_action,
    free_sector,
    freeze_policy,
    freeze_profile,
    is_live,
    policy_action,
    policy_from_solution,
    reward,
)

# ===================================================================
# Def 11-12: Freezing / hashing
# ===================================================================


class TestFreezePolicy:
    def test_deterministic(self):
        p: Policy = {(-1, -1): Coord(0, 0), (-1, 0): Coord(1, 0)}
        assert freeze_policy(p) == freeze_policy(p)

    def test_hashable(self):
        p: Policy = {(-1, -1): Coord(0, 0)}
        f = freeze_policy(p)
        {f}  # must be hashable

    def test_none_action_preserved(self):
        p: Policy = {(-1,): None}
        f = freeze_policy(p)
        assert f == (((-1,), None),)

    def test_sorted_by_state_key(self):
        p: Policy = {(0, 1): Coord(0, 0), (-1, -1): Coord(1, 0)}
        f = freeze_policy(p)
        assert f[0][0] == (-1, -1)
        assert f[1][0] == (0, 1)

    def test_empty_policy(self):
        assert freeze_policy({}) == ()


class TestFreezeProfile:
    def test_two_policies(self):
        p0: Policy = {(-1, -1): Coord(0, 0)}
        p1: Policy = {(-1, -1): Coord(1, 0)}
        fp = freeze_profile([p0, p1])
        assert len(fp) == 2

    def test_identical_policies_distinguishable(self):
        p: Policy = {(-1,): Coord(0, 0)}
        fp = freeze_profile([p, p])
        assert fp[0] == fp[1]

    def test_different_policies_differ(self):
        p0: Policy = {(-1,): Coord(0, 0)}
        p1: Policy = {(-1,): Coord(1, 0)}
        fp = freeze_profile([p0, p1])
        assert fp[0] != fp[1]


# ===================================================================
# Def 13: policy_from_solution
# ===================================================================


class TestPolicyFromSolution:
    def test_single_round(self):
        owner_by_round = ((-1, -1, -1, -1),)
        actions_by_round = ((Coord(0, 0), Coord(1, 0)),)
        pol = policy_from_solution(owner_by_round, actions_by_round, agent_id=0)
        assert pol[(-1, -1, -1, -1)] == Coord(0, 0)
        pol1 = policy_from_solution(owner_by_round, actions_by_round, agent_id=1)
        assert pol1[(-1, -1, -1, -1)] == Coord(1, 0)

    def test_multi_round_trace(self):
        """Two-round trace: policy maps each visited state to an action."""
        owner_by_round = (
            (-1, -1, -1, -1),
            (0, 1, -1, -1),
        )
        actions_by_round = (
            (Coord(0, 0), Coord(1, 0)),
            (Coord(0, 1), Coord(1, 1)),
        )
        pol0 = policy_from_solution(owner_by_round, actions_by_round, agent_id=0)
        assert len(pol0) == 2
        assert pol0[(-1, -1, -1, -1)] == Coord(0, 0)
        assert pol0[(0, 1, -1, -1)] == Coord(0, 1)

    def test_overwrites_duplicate_state(self):
        """If the same state appears twice, the later entry wins."""
        owner_by_round = (
            (-1, -1),
            (-1, -1),  # same state repeated (hypothetical)
        )
        actions_by_round = (
            (Coord(0, 0),),
            (Coord(1, 0),),
        )
        pol = policy_from_solution(owner_by_round, actions_by_round, agent_id=0)
        assert pol[(-1, -1)] == Coord(1, 0)


# ===================================================================
# Def 14: Reward / payoff
# ===================================================================


class TestReward:
    def test_zero_at_start(self, state_2x2: State):
        assert reward(state_2x2, 0) == 0

    def test_equals_largest_region_size(self, state_2x2: State):
        s = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        s = evolve(s, {0: Coord(0, 1), 1: Coord(1, 1)})
        assert reward(s, 0) == 2
        assert reward(s, 1) == 2

    def test_counts_only_largest_component(self):
        """Two disjoint owned sectors: reward is 1 (not 2)."""
        t = Territory.from_ascii(["..", ".."])
        s = State.initial(t, num_agents=1)
        s = evolve(s, {0: Coord(0, 0)})
        s = evolve(s, {0: Coord(1, 1)})
        assert reward(s, 0) == 1


# ===================================================================
# Fallback action
# ===================================================================


class TestFallbackAction:
    def test_distinct_per_agent(self, state_2x2: State):
        a0 = fallback_action(state_2x2, 0)
        a1 = fallback_action(state_2x2, 1)
        assert a0 is not None and a1 is not None
        assert a0 != a1

    def test_returns_none_when_not_enough_sectors(self):
        t = Territory.from_ascii(["."])
        s = State.initial(t, num_agents=2)
        # Only one sector; agent 0 gets it, agent 1 gets None
        assert fallback_action(s, 0) == Coord(0, 0)
        assert fallback_action(s, 1) is None

    def test_invalid_agent_raises(self, state_2x2: State):
        with pytest.raises(ValueError, match="out of range"):
            fallback_action(state_2x2, -1)
        with pytest.raises(ValueError, match="out of range"):
            fallback_action(state_2x2, 2)

    def test_respects_ordering_after_claims(self, state_2x2: State):
        s = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        # Two sectors left: (0,1) and (1,1) in order
        assert fallback_action(s, 0) == Coord(0, 1)
        assert fallback_action(s, 1) == Coord(1, 1)


# ===================================================================
# policy_action
# ===================================================================


class TestPolicyAction:
    def test_uses_policy_entry(self, state_2x2: State):
        key = tuple(state_2x2.owner_by_index)
        policy: Policy = {key: Coord(1, 1)}
        assert policy_action(policy, state_2x2, agent_id=0) == Coord(1, 1)

    def test_falls_back_when_missing(self, state_2x2: State):
        policy: Policy = {}
        assert policy_action(policy, state_2x2, agent_id=0) == fallback_action(
            state_2x2, 0
        )

    def test_falls_back_when_action_no_longer_legal(self, state_2x2: State):
        """If the stored action is owned, fall back."""
        s = evolve(state_2x2, {0: Coord(0, 0), 1: Coord(1, 0)})
        key = tuple(s.owner_by_index)
        # Policy says claim (0,0), but it's already owned
        policy: Policy = {key: Coord(0, 0)}
        act = policy_action(policy, s, agent_id=0)
        assert act == fallback_action(s, 0)

    def test_none_in_policy_triggers_fallback(self, state_2x2: State):
        """A None value in the policy triggers the fallback (None != 'no sector available')."""
        key = tuple(state_2x2.owner_by_index)
        policy: Policy = {key: None}
        act = policy_action(policy, state_2x2, agent_id=0)
        assert act == fallback_action(state_2x2, 0)


# ===================================================================
# Plays, live states and the canonical completion (Definitions 4.2.5, 4.4.2)
# ===================================================================

RUNNING = ["..#.", "#...", ".#.#", "...#", "#.##"]
FIG_3_1 = {
    0: [11, 10, 5, 1],
    1: [8, 7, 12, 9],
    2: [3, 2, 6, 4],
}  # thesis sector numbers


def _fig_3_1_play() -> Play:
    acts = [[FIG_3_1[a][r] - 1 for a in range(3)] for r in range(4)]
    return Play.from_actions(12, 3, acts)


class TestLiveAndDefault:
    def test_free_sector_counts_unoccupied_before(self):
        s = (0, -1, 1, -1, -1)
        assert free_sector(s, 0) == 1
        assert free_sector(s, 1) == 3
        assert free_sector(s, 2) == 4
        assert free_sector(s, 3) is None

    def test_live_needs_num_agents_unoccupied(self):
        assert is_live((-1, -1, 0), 2)
        assert not is_live((-1, 1, 0), 2)

    def test_default_action_only_at_live_states(self):
        with pytest.raises(ValueError, match="live"):
            default_action((-1, 1, 0), 0, 2)

    def test_default_actions_distinct(self):
        s = (-1, 0, -1, -1, 1, -1)
        acts = [default_action(s, a, 3) for a in range(3)]
        assert acts == [0, 2, 3]


class TestPlay:
    def test_from_actions_running_example(self):
        play = _fig_3_1_play()
        assert play.horizon == 4
        assert len(play.states) == 5
        assert all(o != -1 for o in play.final_state)
        assert play.claims_of(0) == (10, 9, 4, 0)

    def test_states_have_t_times_n_occupied(self):
        play = _fig_3_1_play()
        for t, s in enumerate(play.states):
            assert sum(1 for o in s if o != -1) == 3 * t

    def test_rewards_example_3_9_1(self):
        """Example 3.9.1: rewards (2, 4, 4) in s_4."""
        t = Territory.from_ascii(RUNNING)
        assert _fig_3_1_play().rewards(t.neighbor_map()) == (2, 4, 4)

    def test_collision_raises(self):
        with pytest.raises(CollisionError):
            Play.from_actions(4, 2, [[0, 0]])

    def test_occupied_claim_raises(self):
        with pytest.raises(ValueError, match="occupied"):
            Play.from_actions(4, 2, [[0, 1], [0, 2]])

    def test_hashable_and_equal_by_value(self):
        assert _fig_3_1_play() == _fig_3_1_play()
        assert len({_fig_3_1_play(), _fig_3_1_play()}) == 1

    def test_coord_round_trip(self):
        t = Territory.from_ascii(RUNNING)
        play = _fig_3_1_play()
        assert Play.from_coord_actions(t, 3, play.to_coords(t)) == play


class TestCanonicalCompletion:
    def test_recorded_actions_on_play(self):
        play = _fig_3_1_play()
        comp = canonical_completion(play)
        for t in range(play.horizon):
            assert comp.joint_action(play.states[t]) == play.actions[t]

    def test_default_off_play(self):
        comp = canonical_completion(_fig_3_1_play())
        off = (-1,) * 6 + (0,) + (-1,) * 5  # agent 0 holds sector 7 only
        assert not comp.is_on_play(off)
        assert comp.action(off, 1) == 1  # second unoccupied sector: sector 2
        assert comp.action(off, 2) == 2  # third unoccupied sector: sector 3

    def test_example_4_7_1_iteration_0_deviation(self):
        """Example 4.7.1: a claims 7 first; b and c then claim 2, 5, 9 and 4, 6, 12."""
        play = _fig_3_1_play()
        comp = CanonicalCompletion(play)
        state = tuple([-1] * 12)
        dev = [6, 9, 10, 0]  # thesis sectors 7, 10, 11, 1
        b_claims, c_claims = [], []
        for t in range(4):
            prof = [dev[t]] + [
                comp.action(state, b) if t > 0 else play.actions[0][b] for b in (1, 2)
            ]
            if t > 0:
                b_claims.append(prof[1] + 1)
                c_claims.append(prof[2] + 1)
            nxt = list(state)
            for a, i in enumerate(prof):
                assert nxt[i] == -1
                nxt[i] = a
            state = tuple(nxt)
        assert b_claims == [2, 5, 9]
        assert c_claims == [4, 6, 12]
        t = Territory.from_ascii(RUNNING)
        from rcmas.model import rewards_from_owner

        assert rewards_from_owner(state, 3, t.neighbor_map()) == (3, 2, 3)

    def test_undefined_at_non_live_state(self):
        comp = canonical_completion(_fig_3_1_play())
        final = _fig_3_1_play().final_state
        with pytest.raises(ValueError):
            comp.action(final, 0)

    def test_policy_maps_play_states(self):
        t = Territory.from_ascii(RUNNING)
        play = _fig_3_1_play()
        pol = canonical_completion(play).policy(t, 0)
        assert len(pol) == 4
        assert pol[play.states[0]] == t.ordered_sectors()[10]
