"""Definitions 16-23: SMT constraint functions.

Each function adds one class of constraints to a Z3 Optimize instance,
following the paper's numbered definitions.  Every function takes an
``Optimize`` and an ``SmtVariables`` dataclass and mutates the solver
in-place.
"""

from __future__ import annotations

from typing import Any

from .model import Coord, Territory
from .smt_variables import SmtVariables
from .strategy import ActionCandidates, Policy, StateKey


# ---------------------------------------------------------------------------
# Def 16: Initial-state constraint  [Init]
# ---------------------------------------------------------------------------

def init_constraint(opt: Any, v: SmtVariables) -> None:
    """All sectors start unowned at t=0."""
    for s in range(v.S):
        opt.add(v.owner[s][0] == -1)


# ---------------------------------------------------------------------------
# Def 17: Evolution constraint  [Evol]
# ---------------------------------------------------------------------------

def evolution_constraint(opt: Any, v: SmtVariables) -> None:
    """Sector ownership evolves: claimed sectors update, others persist."""
    from z3 import If, Sum

    for t in range(v.T):
        for s in range(v.S):
            claim_sum = Sum([If(v.action[a][t] == s, a + 1, 0) for a in range(v.A)])
            opt.add(v.owner[s][t + 1] == If(claim_sum == 0, v.owner[s][t], claim_sum - 1))


# ---------------------------------------------------------------------------
# Def 18: Protocol constraint  [Prot]  (action domain + availability)
# ---------------------------------------------------------------------------

def protocol_constraint(opt: Any, v: SmtVariables) -> None:
    """Actions must target valid sector indices; can only claim unowned sectors."""
    from z3 import And, Implies

    # Domain: owner in [-1, A-1], action in [0, S-1]
    for s in range(v.S):
        for t in range(v.T + 1):
            opt.add(And(v.owner[s][t] >= -1, v.owner[s][t] < v.A))

    for a in range(v.A):
        for t in range(v.T):
            opt.add(And(v.action[a][t] >= 0, v.action[a][t] < v.S))

    # Availability: can only claim unowned sectors
    for t in range(v.T):
        for a in range(v.A):
            for s in range(v.S):
                opt.add(Implies(v.action[a][t] == s, v.owner[s][t] == -1))


# ---------------------------------------------------------------------------
# Def 19: Collision constraint
# ---------------------------------------------------------------------------

def collision_constraint(opt: Any, v: SmtVariables) -> None:
    """No two agents may claim the same sector in the same round."""
    from z3 import If

    for t in range(v.T):
        for a1 in range(v.A):
            for a2 in range(a1 + 1, v.A):
                opt.add(
                    If(
                        v.action[a1][t] >= 0,
                        If(v.action[a2][t] >= 0, v.action[a1][t] != v.action[a2][t], True),
                        True,
                    )
                )


# ---------------------------------------------------------------------------
# Def 20: Adjacency constraint
# ---------------------------------------------------------------------------

def adjacency_constraint(opt: Any, v: SmtVariables) -> None:
    """adj(i,j,a) iff sectors i,j are physically adjacent and both owned by a at final time."""
    from z3 import And

    final_t = v.T
    for a in range(v.A):
        for i in range(v.S):
            for j in range(i + 1, v.S):
                opt.add(
                    v.get_adj(i, j, a) == And(
                        v.is_phys_adj(i, j),
                        v.owner[i][final_t] == a,
                        v.owner[j][final_t] == a,
                    )
                )


# ---------------------------------------------------------------------------
# Def 21: Cohesive-region (reachability) constraint
# ---------------------------------------------------------------------------

def cohesive_region_constraint(opt: Any, v: SmtVariables) -> None:
    """cr(i,j,a) iff i and j are in the same connected region owned by a.

    Uses interval-based dynamic programming over index span to avoid cycles.
    """
    from z3 import And, Or

    for a in range(v.A):
        for span in range(1, v.S):
            for i in range(v.S - span):
                j = i + span
                chain_terms = []
                for k in range(i + 1, j):
                    chain_terms.append(And(v.get_adj(i, k, a), v.get_cr(k, j, a)))
                    chain_terms.append(And(v.get_cr(i, k, a), v.get_adj(k, j, a)))
                opt.add(v.get_cr(i, j, a) == Or([v.get_adj(i, j, a)] + chain_terms))


# ---------------------------------------------------------------------------
# Def 22: Size constraint
# ---------------------------------------------------------------------------

def size_constraint(opt: Any, v: SmtVariables) -> None:
    """size[i][a] = number of sectors in the connected region rooted at i for agent a."""
    from z3 import If, Sum

    final_t = v.T
    for i in range(v.S):
        for a in range(v.A):
            connections = Sum([If(v.get_cr(i, j, a), 1, 0) for j in range(i + 1, v.S)])
            opt.add(v.size[i][a] == If(v.owner[i][final_t] == a, 1 + connections, 0))


# ---------------------------------------------------------------------------
# Def 23: Reward (payoff) constraint
# ---------------------------------------------------------------------------

def reward_constraint(opt: Any, v: SmtVariables) -> None:
    """payoff[a] = max over all seeds of size[i][a]."""
    from z3 import Or

    for a in range(v.A):
        opt.add(v.payoff[a] >= 0)
        for i in range(v.S):
            opt.add(v.payoff[a] >= v.size[i][a])
        opt.add(Or([v.payoff[a] == v.size[i][a] for i in range(v.S)]))


# ---------------------------------------------------------------------------
# Additional constraints used by IBIS / Q-IBIS (not paper definitions,
# but structural constraints needed for the best-response loop)
# ---------------------------------------------------------------------------

def victory_constraint(opt: Any, v: SmtVariables) -> None:
    """All sectors must be claimed at the final timestep."""
    for s in range(v.S):
        opt.add(v.owner[s][v.T] != -1)


def fixed_actions_constraint(
    opt: Any,
    v: SmtVariables,
    territory: Territory,
    fixed_actions_by_round: tuple[tuple[Coord | None, ...], ...],
) -> None:
    """Fix specific actions for specific agents at specific rounds."""
    T, A = v.T, v.A
    if len(fixed_actions_by_round) != T:
        raise ValueError("fixed_actions_by_round must have length == horizon")
    for t in range(T):
        step = fixed_actions_by_round[t]
        if len(step) != A:
            raise ValueError("fixed_actions_by_round[t] must have length == num_agents")
        for a in range(A):
            c = step[a]
            if c is None:
                continue
            sidx = territory.index_of(c)
            if sidx is None:
                raise ValueError(f"fixed action {c} is not in territory")
            opt.add(v.action[a][t] == sidx)


def fixed_policy_constraint(
    opt: Any,
    v: SmtVariables,
    territory: Territory,
    fixed_policy_by_agent: tuple[Policy | None, ...],
    enforce_state_only_for_agents: tuple[int, ...] = (),
) -> None:
    """Constrain agents to follow fixed state-only policies.

    Agents with a non-None policy have their actions determined by a
    combination of explicit state->action mappings and a deterministic
    fallback for unmapped states.  Agents listed in
    *enforce_state_only_for_agents* additionally have same-state =>
    same-action consistency enforced across the horizon.
    """
    from z3 import And, If, Implies, Sum

    S, A, T = v.S, v.A, v.T

    if len(fixed_policy_by_agent) != A:
        raise ValueError("fixed_policy_by_agent must have length == num_agents")

    for a in enforce_state_only_for_agents:
        if a < 0 or a >= A:
            raise ValueError("enforce_state_only_for_agents contains invalid agent index")

    def default_action_expr(agent_id: int, t: int):
        expr = -1
        for i in reversed(range(S)):
            unowned_before = Sum([If(v.owner[j][t] == -1, 1, 0) for j in range(i)])
            cond = And(v.owner[i][t] == -1, unowned_before == agent_id)
            expr = If(cond, i, expr)
        return expr

    # Validate policies
    for a in range(A):
        policy = fixed_policy_by_agent[a]
        if policy is None:
            continue
        for state, chosen_coord in policy.items():
            if len(state) != S:
                raise ValueError("fixed_policy_by_agent has a state key with wrong length")
            if chosen_coord is not None:
                sidx = territory.index_of(chosen_coord)
                if sidx is None:
                    raise ValueError(f"fixed policy action {chosen_coord} is not in territory")

    # Apply policy constraints
    for t in range(T):
        for a in range(A):
            policy = fixed_policy_by_agent[a]
            if policy is None:
                continue

            expr = default_action_expr(a, t)

            for state in sorted(policy.keys()):
                chosen_coord = policy[state]
                chosen_idx = -1
                if chosen_coord is not None:
                    sidx = territory.index_of(chosen_coord)
                    if sidx is None:
                        raise ValueError(f"fixed policy action {chosen_coord} is not in territory")
                    chosen_idx = sidx

                match_state = And([v.owner[s][t] == state[s] for s in range(S)])
                expr = If(match_state, chosen_idx, expr)

            opt.add(v.action[a][t] == expr)

    # State-only consistency
    for a in enforce_state_only_for_agents:
        for t1 in range(T):
            for t2 in range(t1 + 1, T):
                same_state = And([v.owner[s][t1] == v.owner[s][t2] for s in range(S)])
                opt.add(Implies(same_state, v.action[a][t1] == v.action[a][t2]))


def action_candidates_constraint(
    opt: Any,
    v: SmtVariables,
    territory: Territory,
    action_candidates_by_agent: tuple[ActionCandidates | None, ...],
) -> None:
    """Restrict agent actions to RL-proposed candidate sets per visited state."""
    from z3 import And, Implies, Or

    S, A, T = v.S, v.A, v.T

    if len(action_candidates_by_agent) != A:
        raise ValueError("action_candidates_by_agent must have length == num_agents")

    # Validate
    for a in range(A):
        cand = action_candidates_by_agent[a]
        if cand is None:
            continue
        for state, choices in cand.items():
            if len(state) != S:
                raise ValueError("action_candidates_by_agent has a state key with wrong length")
            for c in choices:
                if c is not None and territory.index_of(c) is None:
                    raise ValueError(f"candidate action {c} is not in territory")

    # Apply restrictions
    for t in range(T):
        for a in range(A):
            cand = action_candidates_by_agent[a]
            if cand is None:
                continue

            for state in sorted(cand.keys()):
                allowed_idxs: set[int] = {-1}
                for c in cand[state]:
                    if c is None:
                        allowed_idxs.add(-1)
                        continue
                    sidx = territory.index_of(c)
                    if sidx is None:
                        raise ValueError(f"candidate action {c} is not in territory")
                    allowed_idxs.add(sidx)

                match_state = And([v.owner[s][t] == state[s] for s in range(S)])
                opt.add(Implies(match_state, Or([v.action[a][t] == i for i in sorted(allowed_idxs)])))
