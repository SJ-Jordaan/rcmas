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
from .symmetry import ComponentInfo, LocalSymmetryInfo, SymmetryInfo


# ---------------------------------------------------------------------------
# Def 16: Initial-state constraint  [Init]
# ---------------------------------------------------------------------------

def init_constraint(opt: Any, v: SmtVariables) -> None:
    """All sectors start unowned at t=0."""
    opt.add(*[v.owner[s][0] == -1 for s in range(v.S)])


# ---------------------------------------------------------------------------
# Def 17: Evolution constraint  [Evol]
# ---------------------------------------------------------------------------

def evolution_constraint(opt: Any, v: SmtVariables) -> None:
    """Sector ownership evolves: claimed sectors update, others persist."""
    from z3 import If, Sum

    constraints = []
    for t in range(v.T):
        for s in range(v.S):
            claim_sum = Sum([If(v.action[a][t] == s, a + 1, 0) for a in range(v.A)])
            constraints.append(v.owner[s][t + 1] == If(claim_sum == 0, v.owner[s][t], claim_sum - 1))
    opt.add(*constraints)


# ---------------------------------------------------------------------------
# Def 18: Protocol constraint  [Prot]  (action domain + availability)
# ---------------------------------------------------------------------------

def protocol_constraint(opt: Any, v: SmtVariables) -> None:
    """Actions must target valid sector indices; can only claim unowned sectors."""
    from z3 import And, Implies

    constraints = []
    # Domain: owner in [-1, A-1], action in [0, S-1]
    for s in range(v.S):
        for t in range(v.T + 1):
            constraints.append(And(v.owner[s][t] >= -1, v.owner[s][t] < v.A))

    for a in range(v.A):
        for t in range(v.T):
            constraints.append(And(v.action[a][t] >= 0, v.action[a][t] < v.S))

    # Availability: can only claim unowned sectors
    for t in range(v.T):
        for a in range(v.A):
            for s in range(v.S):
                constraints.append(Implies(v.action[a][t] == s, v.owner[s][t] == -1))
    opt.add(*constraints)


# ---------------------------------------------------------------------------
# Def 19: Collision constraint
# ---------------------------------------------------------------------------

def collision_constraint(opt: Any, v: SmtVariables) -> None:
    """No two agents may claim the same sector in the same round."""
    if v.A >= 3:
        # Distinct's all-different propagator amortises A*(A-1)/2 pairwise
        # checks into O(A) arc-consistency passes — net win for A >= 3.
        from z3 import Distinct

        opt.add(*[
            Distinct([v.action[a][t] for a in range(v.A)])
            for t in range(v.T)
        ])
    else:
        # For A <= 2, pairwise != is a single constraint per round with no
        # theory-propagator overhead.
        opt.add(*[
            v.action[a1][t] != v.action[a2][t]
            for t in range(v.T)
            for a1 in range(v.A)
            for a2 in range(a1 + 1, v.A)
        ])


# ---------------------------------------------------------------------------
# Def 20: Adjacency constraint
# ---------------------------------------------------------------------------

def adjacency_constraint(opt: Any, v: SmtVariables) -> None:
    """adj(i,j,a) iff sectors i,j are physically adjacent and both owned by a at final time."""
    from z3 import And

    final_t = v.T
    # Only constrain adj variables that exist (physically adjacent pairs)
    constraints = []
    for (i, j, a), adj_var in v.adj.items():
        constraints.append(
            adj_var == And(
                v.owner[i][final_t] == a,
                v.owner[j][final_t] == a,
            )
        )
    if constraints:
        opt.add(*constraints)


# ---------------------------------------------------------------------------
# Def 21: Cohesive-region (reachability) constraint
# ---------------------------------------------------------------------------

def cohesive_region_constraint(opt: Any, v: SmtVariables) -> None:
    """cr(i,j,a) iff i and j are in the same connected region owned by a.

    Uses interval-based dynamic programming over index span to avoid cycles.
    Chain terms involving non-adjacent or pruned pairs are eliminated at
    construction time (get_adj/get_cr return Python False, collapsing terms).
    Constraints are only generated for cr variables that exist (unpruned).
    """
    from z3 import And, Or

    constraints = []
    for a in range(v.A):
        for span in range(1, v.S):
            for i in range(v.S - span):
                j = i + span
                cr_ij = v.get_cr(i, j, a)
                # Skip if this cr variable was pruned (unreachable pair)
                if cr_ij is False:
                    continue
                adj_ij = v.get_adj(i, j, a)
                disjuncts = [adj_ij] if adj_ij is not False else []
                for k in range(i + 1, j):
                    adj_ik = v.get_adj(i, k, a)
                    adj_kj = v.get_adj(k, j, a)
                    cr_kj = v.get_cr(k, j, a)
                    cr_ik = v.get_cr(i, k, a)
                    if adj_ik is not False and cr_kj is not False:
                        disjuncts.append(And(adj_ik, cr_kj))
                    if cr_ik is not False and adj_kj is not False:
                        disjuncts.append(And(cr_ik, adj_kj))
                if disjuncts:
                    constraints.append(cr_ij == Or(disjuncts))
                else:
                    constraints.append(cr_ij == False)
    if constraints:
        opt.add(*constraints)


# ---------------------------------------------------------------------------
# Def 22: Size constraint
# ---------------------------------------------------------------------------

def size_constraint(opt: Any, v: SmtVariables) -> None:
    """size[i][a] = weighted size of the connected region rooted at i for agent a.

    When ``v.weights`` is ``None`` every sector has unit weight (original
    behaviour).  Otherwise ``v.weights[i]`` gives the weight of sector *i*
    and the size sums the weights of connected sectors.
    """
    from z3 import If, Sum

    final_t = v.T
    constraints = []
    for i in range(v.S):
        w_i = v.weights[i] if v.weights is not None else 1
        for a in range(v.A):
            connections = Sum([
                If(v.get_cr(i, j, a), v.weights[j] if v.weights is not None else 1, 0)
                for j in range(i + 1, v.S)
            ])
            constraints.append(v.size[i][a] == If(v.owner[i][final_t] == a, w_i + connections, 0))
    opt.add(*constraints)


# ---------------------------------------------------------------------------
# Def 23: Reward (payoff) constraint
# ---------------------------------------------------------------------------

def reward_constraint(opt: Any, v: SmtVariables) -> None:
    """payoff[a] = max over all seeds of size[i][a]."""
    from z3 import Or

    constraints = []
    for a in range(v.A):
        constraints.append(v.payoff[a] >= 0)
        for i in range(v.S):
            constraints.append(v.payoff[a] >= v.size[i][a])
        constraints.append(Or([v.payoff[a] == v.size[i][a] for i in range(v.S)]))
    opt.add(*constraints)


# ---------------------------------------------------------------------------
# Additional constraints used by IBIS / Q-IBIS (not paper definitions,
# but structural constraints needed for the best-response loop)
# ---------------------------------------------------------------------------

def victory_constraint(opt: Any, v: SmtVariables) -> None:
    """All sectors must be claimed at the final timestep."""
    opt.add(*[v.owner[s][v.T] != -1 for s in range(v.S)])


def weight_balance_constraint(opt: Any, v: SmtVariables, target_weight: int) -> None:
    """Each agent's total owned weight must equal *target_weight* at the final timestep.

    Used by the abstract RCMAS encoding to ensure that each agent's
    assigned blocks sum to exactly ``|T_concrete| / num_agents`` concrete
    sectors, preserving the RCMAS demand invariant.

    Uses Z3's ``PbEq`` (pseudo-boolean equality) for specialised cardinality
    reasoning when available, falling back to ``Sum``/``If`` otherwise.
    """
    final_t = v.T
    assert v.weights is not None

    try:
        from z3 import PbEq
        opt.add(*[
            PbEq(
                [(v.owner[s][final_t] == a, v.weights[s]) for s in range(v.S)],
                target_weight,
            )
            for a in range(v.A)
        ])
    except ImportError:
        from z3 import If, Sum
        for a in range(v.A):
            total = Sum([
                If(v.owner[s][final_t] == a, v.weights[s], 0)
                for s in range(v.S)
            ])
            opt.add(total == target_weight)


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


# ---------------------------------------------------------------------------
# Symmetry-breaking constraints
# ---------------------------------------------------------------------------

def symmetry_breaking_constraint(
    opt: Any,
    v: SmtVariables,
    sym_info: SymmetryInfo,
    demands: tuple[int, ...] | None = None,
) -> None:
    """Add symmetry-breaking constraints to prune equivalent solutions.

    Two types of constraint are added:

    1. **Agent lex-leader**: within each demand class, first-round actions
       are ordered by agent index.  When *demands* is ``None``, all agents
       are assumed to share a common demand (global ordering, factor A!).
       With heterogeneous demands, only agents in the same demand class
       are interchangeable (factor ∏|C|!).

    2. **Spatial canonicalization**: the first agent's first action is
       restricted to one canonical representative per territory-automorphism
       orbit.  This eliminates spatial symmetry (factor |Aut(T)|).
    """
    from z3 import Or

    A = v.A

    # Agent lex-leader on round-0 actions (within demand classes)
    if demands is None:
        # Uniform demands assumed — global S_n ordering
        for a in range(A - 1):
            opt.add(v.action[a][0] <= v.action[a + 1][0])
    else:
        from .symmetry import demand_classes

        for cls in demand_classes(demands):
            for i in range(len(cls) - 1):
                opt.add(v.action[cls[i]][0] <= v.action[cls[i + 1]][0])

    # Spatial canonicalization: first agent picks from orbit representatives
    reps = sym_info.representatives
    if reps:
        opt.add(Or([v.action[0][0] == r for r in reps]))


def local_symmetry_breaking_constraint(
    opt: Any,
    v: SmtVariables,
    local_sym: LocalSymmetryInfo,
) -> None:
    """Add conditional local symmetry-breaking constraints (Ch5 §3).

    For each non-identity local automorphism σ_R with problematic boundary
    pairs BP(σ_R), adds:

        (∧_{(i,j)∈BP} ¬(sec^i_m = sec^j_m ∧ sec^i_m ≠ -1))
            ⟹ (∀a: act^a_0 ∈ R ⟹ act^a_0 ∈ Reps_R)

    The antecedent checks reward compatibility at the terminal state.
    The consequent restricts first actions within R to orbit representatives.
    """
    from z3 import And, Implies, Not, Or

    A, S = v.A, v.S
    final_t = v.T
    reps = set(local_sym.representatives)
    subregion = local_sym.subregion

    if not reps or not local_sym.boundary_pairs:
        return

    # Collect all unique boundary pairs across all automorphisms
    all_bp: set[tuple[int, int]] = set()
    for bp_set in local_sym.boundary_pairs.values():
        all_bp.update(bp_set)

    if not all_bp:
        return

    # Antecedent: no agent spans a problematic boundary pair at terminal state
    compat_clauses = []
    for i, j in sorted(all_bp):
        # ¬(sec^i_m = sec^j_m ∧ sec^i_m ≠ -1)
        compat_clauses.append(
            Not(And(v.owner[i][final_t] == v.owner[j][final_t],
                    v.owner[i][final_t] != -1))
        )
    antecedent = And(compat_clauses) if len(compat_clauses) > 1 else compat_clauses[0]

    # Consequent: for each agent, if first action is in R, it must be an orbit rep
    consequent_clauses = []
    non_rep_in_r = sorted(subregion - reps)
    for a in range(A):
        for s in non_rep_in_r:
            # act^a_0 ≠ s (forbid non-representative sectors in R)
            consequent_clauses.append(v.action[a][0] != s)

    if not consequent_clauses:
        return

    consequent = And(consequent_clauses) if len(consequent_clauses) > 1 else consequent_clauses[0]
    opt.add(Implies(antecedent, consequent))


def disconnected_symmetry_constraint(
    opt: Any,
    v: SmtVariables,
    comp_info: ComponentInfo,
    demands: tuple[int, ...] | None = None,
) -> None:
    """Add symmetry-breaking constraints for disconnected territories (Ch5 §4).

    Three types of constraints:
    1. Agent lex-leader within demand classes (reuses symmetry_breaking_constraint logic)
    2. Intra-component spatial canonicalization (orbit reps within each component)
    3. Inter-component ordering for isomorphic components
    """
    from z3 import And, If, Implies, Or

    A = v.A

    # 1. Agent lex-leader (within demand classes)
    if demands is None:
        for a in range(A - 1):
            opt.add(v.action[a][0] <= v.action[a + 1][0])
    else:
        from .symmetry import demand_classes

        for cls in demand_classes(demands):
            for i in range(len(cls) - 1):
                opt.add(v.action[cls[i]][0] <= v.action[cls[i + 1]][0])

    # 2. Intra-component spatial canonicalization
    # For each component, restrict first agent's first action in that component
    # to orbit representatives within that component
    for comp_idx, comp in enumerate(comp_info.components):
        reps = set(comp_info.component_representatives[comp_idx])
        non_rep = sorted(comp - reps)
        if not non_rep:
            continue
        # First agent (agent 0): if action is in this component, must be a rep
        for s in non_rep:
            opt.add(v.action[0][0] != s)

    # 3. Inter-component ordering for isomorphic components
    # For each iso class with ≥2 components, encode:
    #   agent 0's first action in the lowest-indexed component is preferred.
    # Sound constraint: for each iso class, the minimum first-round action
    # (over all agents) in component K_l must be ≤ the minimum in K_l'
    # when l < l' (under canonical labeling within the component).
    #
    # Encoding: for each pair of isomorphic components (K_l, K_l'),
    # for all agents a, b: (act^a_0 ∈ K_l' ∧ act^b_0 ∈ K_l)
    #   ⟹ rank_within_component(act^b_0 in K_l) ≤ rank(act^a_0 in K_l')
    #
    # Simplified: use the isomorphism witness to map K_l' sector indices
    # to K_l indices, then compare within the same index space.
    for iso_class in comp_info.iso_classes:
        if len(iso_class) < 2:
            continue

        for idx in range(len(iso_class) - 1):
            ci_first = iso_class[idx]
            ci_second = iso_class[idx + 1]
            comp_first = comp_info.components[ci_first]
            comp_second = comp_info.components[ci_second]

            # Build rank within each component (position in sorted order)
            first_sorted = sorted(comp_first)
            second_sorted = sorted(comp_second)
            rank_first = {s: r for r, s in enumerate(first_sorted)}
            rank_second = {s: r for r, s in enumerate(second_sorted)}

            # For all agents a, b:
            #   (act^a_0 ∈ K_first ∧ act^b_0 ∈ K_second)
            #   ⟹ rank(act^a_0) ≤ rank(act^b_0)
            for a in range(A):
                for b in range(A):
                    for sf in first_sorted:
                        rf = rank_first[sf]
                        for ss in second_sorted:
                            rs = rank_second[ss]
                            if rf > rs:
                                # This pairing violates component ordering
                                opt.add(
                                    Implies(
                                        And(v.action[a][0] == sf,
                                            v.action[b][0] == ss),
                                        False
                                    )
                                )
