"""Tests for smt_constraints.py (Def 16-23).

Each constraint is tested in isolation on tiny grids to verify its
individual semantic contribution.  Tests compose only the minimal set
of constraints needed for the property under test.
"""

from __future__ import annotations

import pytest

from rcmas.model import Coord, Territory
from rcmas.smt_variables import SmtVariables, create_variables
from rcmas.smt_constraints import (
    action_candidates_constraint,
    adjacency_constraint,
    cohesive_region_constraint,
    collision_constraint,
    evolution_constraint,
    fixed_policy_constraint,
    init_constraint,
    protocol_constraint,
    reward_constraint,
    size_constraint,
    victory_constraint,
)

z3 = pytest.importorskip("z3")


def _opt() -> "z3.Optimize":
    return z3.Optimize()


def _model_val(m, expr) -> int:
    return m.eval(expr, model_completion=True).as_long()


def _model_bool(m, expr) -> bool:
    if isinstance(expr, bool):
        return expr
    return z3.is_true(m.eval(expr, model_completion=True))


# ===================================================================
# Def 16: Init constraint  [Init]
# ===================================================================

class TestInitConstraint:
    def test_all_unowned_at_t0(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 1)
        opt = _opt()
        init_constraint(opt, v)
        assert opt.check() == z3.sat
        m = opt.model()
        for s in range(v.S):
            assert _model_val(m, v.owner[s][0]) == -1

    def test_multi_sector(self):
        v = create_variables(Territory.from_ascii(["..", ".."]), 2, 1)
        opt = _opt()
        init_constraint(opt, v)
        assert opt.check() == z3.sat
        m = opt.model()
        for s in range(v.S):
            assert _model_val(m, v.owner[s][0]) == -1

    def test_prevents_owned_at_t0(self):
        v = create_variables(Territory.from_ascii(["."]), 1, 1)
        opt = _opt()
        init_constraint(opt, v)
        opt.add(v.owner[0][0] == 0)
        assert opt.check() == z3.unsat


# ===================================================================
# Def 17: Evolution constraint  [Evol]
# ===================================================================

class TestEvolutionConstraint:
    def test_unclaimed_sector_persists(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 1)
        opt = _opt()
        init_constraint(opt, v)
        evolution_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.owner[0][1]) == 0
        assert _model_val(m, v.owner[1][1]) == -1

    def test_claimed_sector_stays_owned(self):
        """Once claimed at t, still owned at t+1 with no further claims."""
        v = create_variables(Territory.from_ascii([".."]), 1, 2)
        opt = _opt()
        init_constraint(opt, v)
        evolution_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[0][1] == 1)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.owner[0][1]) == 0
        assert _model_val(m, v.owner[0][2]) == 0  # still owned
        assert _model_val(m, v.owner[1][2]) == 0  # just claimed

    def test_two_agents_claim_different(self):
        v = create_variables(Territory.from_ascii([".."]), 2, 1)
        opt = _opt()
        init_constraint(opt, v)
        evolution_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[1][0] == 1)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.owner[0][1]) == 0
        assert _model_val(m, v.owner[1][1]) == 1


# ===================================================================
# Def 18: Protocol constraint  [Prot]
# ===================================================================

class TestProtocolConstraint:
    def test_action_domain_is_sector_indices(self):
        """Actions must be in [0, S-1]."""
        v = create_variables(Territory.from_ascii([".."]), 1, 1)
        opt = _opt()
        protocol_constraint(opt, v)
        opt.add(v.action[0][0] == -1)  # invalid: no no-op
        assert opt.check() == z3.unsat

    def test_action_out_of_range_rejected(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 1)
        opt = _opt()
        protocol_constraint(opt, v)
        opt.add(v.action[0][0] == 5)  # only 2 sectors
        assert opt.check() == z3.unsat

    def test_prevents_reclaiming_owned(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[0][1] == 0)  # sector 0 already owned
        assert opt.check() == z3.unsat

    def test_owner_domain_bounded(self):
        """Owner values must be in [-1, A-1]."""
        v = create_variables(Territory.from_ascii(["."]), 2, 1)
        opt = _opt()
        protocol_constraint(opt, v)
        opt.add(v.owner[0][0] == 5)  # invalid
        assert opt.check() == z3.unsat


# ===================================================================
# Def 19: Collision constraint
# ===================================================================

class TestCollisionConstraint:
    def test_same_sector_is_unsat(self):
        v = create_variables(Territory.from_ascii([".."]), 2, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[1][0] == 0)
        assert opt.check() == z3.unsat

    def test_different_sectors_allowed(self):
        v = create_variables(Territory.from_ascii([".."]), 2, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[1][0] == 1)
        assert opt.check() == z3.sat

    def test_three_agent_pairwise(self):
        """Three agents: all pairs must be collision-free."""
        v = create_variables(Territory.from_ascii(["..."]), 3, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[1][0] == 1)
        opt.add(v.action[2][0] == 0)  # collides with agent 0
        assert opt.check() == z3.unsat

    def test_collision_only_within_same_round(self):
        """Agents claiming the same sector in different rounds is fine (protocol prevents re-claim, but no collision)."""
        v = create_variables(Territory.from_ascii(["..."]), 2, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        evolution_constraint(opt, v)
        # Round 0: agent 0 claims 0, agent 1 claims 1
        # Round 1: agent 0 claims 2, agent 1 claims... sector 2?
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[1][0] == 1)
        opt.add(v.action[0][1] == 2)
        opt.add(v.action[1][1] == 2)  # collision at round 1
        assert opt.check() == z3.unsat


# ===================================================================
# Def 20: Adjacency constraint
# ===================================================================

class TestAdjacencyConstraint:
    def test_one_owned_sector_no_adj(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        assert opt.check() == z3.sat
        m = opt.model()
        assert not _model_bool(m, v.get_adj(0, 1, 0))

    def test_both_owned_neighbours_true(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[0][1] == 1)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_bool(m, v.get_adj(0, 1, 0))

    def test_non_adjacent_sectors_false(self):
        """On a 1x3 territory, sectors 0 and 2 are not physically adjacent."""
        v = create_variables(Territory.from_ascii(["..."]), 1, 3)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[0][1] == 2)
        opt.add(v.action[0][2] == 1)
        assert opt.check() == z3.sat
        m = opt.model()
        # After round 1: own 0 and 2 — they are not adjacent
        # (adj is evaluated at final_t = 3, where all are owned, so 0-2 adj depends on physical adjacency)
        assert not _model_bool(m, v.get_adj(0, 2, 0))
        assert _model_bool(m, v.get_adj(0, 1, 0))
        assert _model_bool(m, v.get_adj(1, 2, 0))

    def test_different_owners_false(self):
        v = create_variables(Territory.from_ascii([".."]), 2, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[1][0] == 1)
        assert opt.check() == z3.sat
        m = opt.model()
        assert not _model_bool(m, v.get_adj(0, 1, 0))
        assert not _model_bool(m, v.get_adj(0, 1, 1))


# ===================================================================
# Def 21: Cohesive-region constraint
# ===================================================================

class TestCohesiveRegionConstraint:
    def test_direct_neighbours_connected(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[0][1] == 1)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_bool(m, v.get_cr(0, 1, 0))

    def test_transitive_reachability(self):
        """On 1x3, sectors 0 and 2 are connected through sector 1."""
        v = create_variables(Territory.from_ascii(["..."]), 1, 3)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[0][1] == 1)
        opt.add(v.action[0][2] == 2)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_bool(m, v.get_cr(0, 2, 0))


# ===================================================================
# Def 22: Size constraint
# ===================================================================

class TestSizeConstraint:
    def test_single_owned_size_one(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        size_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.size[0][0]) == 1
        assert _model_val(m, v.size[1][0]) == 0

    def test_two_connected_size(self):
        """size[i][a] = 1 + count of connected sectors with index > i.
        So size[0][0] = 2 (sector 0 + sector 1), size[1][0] = 1 (only itself)."""
        v = create_variables(Territory.from_ascii([".."]), 1, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        size_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[0][1] == 1)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.size[0][0]) == 2  # seed 0 sees sector 1 via cr
        assert _model_val(m, v.size[1][0]) == 1  # seed 1 has no higher-index neighbours

    def test_unowned_sector_size_zero(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        size_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.size[1][0]) == 0


# ===================================================================
# Def 23: Reward constraint
# ===================================================================

class TestRewardConstraint:
    def test_payoff_equals_max_size(self):
        v = create_variables(Territory.from_ascii([".."]), 1, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        size_constraint(opt, v)
        reward_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[0][1] == 1)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.payoff[0]) == 2

    def test_payoff_zero_when_nothing_owned(self):
        """Single-agent claims nothing -> payoff must be 0.
        Since action >= 0 forces a claim, use a 2-agent setup where
        agent 1's payoff is 0 because agent 0 owns the only sector."""
        v = create_variables(Territory.from_ascii(["."]), 1, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        size_constraint(opt, v)
        reward_constraint(opt, v)
        opt.add(v.action[0][0] == 0)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.payoff[0]) == 1

    def test_two_agents_distinct_payoffs(self):
        """On 1x4, 2 agents, 2 rounds: one agent gets connected pair (payoff 2),
        other gets disconnected pair (payoff 1)."""
        v = create_variables(Territory.from_ascii(["...."]), 2, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        size_constraint(opt, v)
        reward_constraint(opt, v)
        # Agent 0 claims 0 then 3 (endpoints, not adjacent on 1x4) -> payoff 1
        # Agent 1 claims 1 then 2 (adjacent) -> payoff 2
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[1][0] == 1)
        opt.add(v.action[0][1] == 3)
        opt.add(v.action[1][1] == 2)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.payoff[0]) == 1
        assert _model_val(m, v.payoff[1]) == 2

    def test_payoff_reflects_largest_component(self):
        """On 2x2, agent owns (0,0) and (1,1) — diagonal, not connected. Payoff = 1."""
        v = create_variables(Territory.from_ascii(["..", ".."]), 2, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        evolution_constraint(opt, v)
        adjacency_constraint(opt, v)
        cohesive_region_constraint(opt, v)
        size_constraint(opt, v)
        reward_constraint(opt, v)
        # Sectors: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
        opt.add(v.action[0][0] == 0)
        opt.add(v.action[1][0] == 1)
        opt.add(v.action[0][1] == 3)  # diagonal
        opt.add(v.action[1][1] == 2)
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.payoff[0]) == 1  # (0,0) and (1,1) not connected
        assert _model_val(m, v.payoff[1]) == 1  # (1,0) and (0,1) not connected


# ===================================================================
# Victory constraint
# ===================================================================

class TestVictoryConstraint:
    def test_unsat_when_insufficient_rounds(self):
        """1 agent, 2 sectors, 1 round -> can only claim 1 -> unsat."""
        v = create_variables(Territory.from_ascii([".."]), 1, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        victory_constraint(opt, v)
        assert opt.check() == z3.unsat

    def test_sat_when_enough_rounds(self):
        """1 agent, 2 sectors, 2 rounds -> can fill board."""
        v = create_variables(Territory.from_ascii([".."]), 1, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        collision_constraint(opt, v)
        victory_constraint(opt, v)
        assert opt.check() == z3.sat

    def test_two_agents_fill_board_in_one_round(self):
        """2 agents on 2 sectors, 1 round: both claim distinct sectors."""
        v = create_variables(Territory.from_ascii([".."]), 2, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        evolution_constraint(opt, v)
        victory_constraint(opt, v)
        assert opt.check() == z3.sat

    def test_three_agents_four_sectors_two_rounds(self):
        """3 agents on 4 sectors need 2 rounds: 3+1=4. But second round only
        needs 1 claim; the other 2 agents must still pick valid (unowned) sectors,
        which don't exist. So this is unsat."""
        v = create_variables(Territory.from_ascii(["..", ".."]), 3, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        evolution_constraint(opt, v)
        victory_constraint(opt, v)
        # Round 1: 3 agents claim 3 of 4 sectors. Round 2: all 3 must claim,
        # but only 1 unowned sector remains -> 2 agents have no valid action.
        assert opt.check() == z3.unsat


# ===================================================================
# Fixed-policy constraint
# ===================================================================

class TestFixedPolicyConstraint:
    def test_forces_specified_action(self):
        """On 2x1, fix agent 0 to claim sector 0 at the initial state."""
        t = Territory.from_ascii(["..", ".."]) # 4 sectors, 2 agents, 2 rounds
        v = create_variables(t, 2, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)
        evolution_constraint(opt, v)

        initial_state = (-1, -1, -1, -1)
        policy_a0 = {initial_state: Coord(0, 0)}
        fixed_policy_constraint(opt, v, t, (policy_a0, None))
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.action[0][0]) == 0

    def test_wrong_state_key_length_raises(self):
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 1)
        opt = _opt()
        bad_policy = {(-1, -1): Coord(0, 0)}  # length 2, should be 4
        with pytest.raises(ValueError, match="wrong length"):
            fixed_policy_constraint(opt, v, t, (bad_policy, None))

    def test_state_only_consistency(self):
        """enforce_state_only_for_agents ensures same state => same action."""
        t = Territory.from_ascii(["..", ".."]) # 4 sectors
        v = create_variables(t, 1, 2)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        evolution_constraint(opt, v)
        # Fix: same state => same action for agent 0 across horizon
        fixed_policy_constraint(opt, v, t, (None,), enforce_state_only_for_agents=(0,))
        # This just adds constraints; the model is still satisfiable
        assert opt.check() == z3.sat


# ===================================================================
# Action-candidates constraint
# ===================================================================

class TestActionCandidatesConstraint:
    def test_restricts_to_candidates(self):
        t = Territory.from_ascii(["..", ".."]) # 4 sectors
        v = create_variables(t, 2, 1)
        opt = _opt()
        init_constraint(opt, v)
        protocol_constraint(opt, v)
        collision_constraint(opt, v)

        # Agent 0 can only claim sector 0 at the initial state
        candidates = {(-1, -1, -1, -1): (Coord(0, 0),)}
        action_candidates_constraint(opt, v, t, (candidates, None))
        assert opt.check() == z3.sat
        m = opt.model()
        assert _model_val(m, v.action[0][0]) == 0

    def test_wrong_length_raises(self):
        t = Territory.from_ascii(["..", ".."])
        v = create_variables(t, 2, 1)
        opt = _opt()
        bad = {(-1, -1): (Coord(0, 0),)}
        with pytest.raises(ValueError, match="wrong length"):
            action_candidates_constraint(opt, v, t, (bad, None))
