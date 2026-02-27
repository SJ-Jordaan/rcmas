"""Tests for strategy.py (Def 9-14)."""

from __future__ import annotations

import pytest

from rcmas.model import Coord, State, Territory, evolve
from rcmas.strategy import (
    Policy,
    fallback_action,
    freeze_policy,
    freeze_profile,
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
        assert policy_action(policy, state_2x2, agent_id=0) == fallback_action(state_2x2, 0)

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
