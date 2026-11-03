"""Constraint families of the encoding of Chapter 4.

Each function adds one constraint family to a Z3 solver (or to any object
with an ``add`` method, such as the collector of :mod:`rcmas.smt_solve`).
The formula of Section 4.2 is

    F = [Init] and [Evol] and [Prot] and [Dom] and [Wit] and [Rew] and [phi_sat]

with :func:`init_constraint` (Definition 4.2.1), :func:`evolution_constraint`
(Definition 4.2.2, whose clause 4 excludes collisions),
:func:`protocol_constraint` (Definition 4.2.3), :func:`domain_constraint`
(Definition 4.2.4), :func:`witness_constraint` (Definition 4.3.1),
:func:`reward_constraint` (Definition 4.3.2) and
:func:`qualitative_constraint` (Definition 4.3.3).  The opponent constraint
``[Fix^a_pi]`` of the best-response formula ``G_a(pi) = F and [Fix^a_pi]``
(Definition 4.7.1) is :func:`opponent_constraint`.

With sector weights and a custom adjacency in the variables, the witness and
reward constraints encode the weighted reward of the abstract RCMAS
(Definition 6.1.2 and Section 6.1): ``reward_a`` is the total weight of the
witness region, and the levels count blocks.

The remaining functions are structural constraints used by Q-IBIS and
CEGAR-NE (fixed policies, candidate sets, weight balance) and the
symmetry-breaking constraints of Chapter 5, which are never part of the
cached formula ``F`` (Section 7.1).

Tool numbering: sectors ``0..|T|-1``, agents ``0..n-1``, ``-1`` for an
unoccupied sector (see :mod:`rcmas.smt_variables`).
"""

from __future__ import annotations

from typing import Any

from .model import Coord, Territory
from .smt_variables import SmtVariables
from .strategy import ActionCandidates, Play, Policy
from .symmetry import ComponentInfo, LocalSymmetryInfo, SymmetryInfo

# ---------------------------------------------------------------------------
# Definition 4.2.1: initial state constraint [Init]
# ---------------------------------------------------------------------------


def init_constraint(opt: Any, v: SmtVariables) -> None:
    """All sectors are unoccupied at step 0."""
    opt.add(*[v.owner[i][0] == -1 for i in range(v.S)])


# ---------------------------------------------------------------------------
# Definition 4.2.2: evolution constraint [Evol]
# ---------------------------------------------------------------------------


def evolution_constraint(opt: Any, v: SmtVariables) -> None:
    """Clauses 1-4 of Definition 4.2.2.

    1. a sector keeps its controller;
    2. an unoccupied sector that no agent claims stays unoccupied;
    3. a claimed sector passes to its claimant;
    4. two agents never claim the same sector in the same round.
    """
    from z3 import And, Implies

    constraints = []
    for t in range(v.T):
        for i in range(v.S):
            for a in range(v.A):
                constraints.append(Implies(v.owner[i][t] == a, v.owner[i][t + 1] == a))
            unclaimed = [v.action[a][t] != i for a in range(v.A)]
            constraints.append(
                Implies(And(v.owner[i][t] == -1, *unclaimed), v.owner[i][t + 1] == -1)
            )
            for a in range(v.A):
                constraints.append(Implies(v.action[a][t] == i, v.owner[i][t + 1] == a))
    if constraints:
        opt.add(*constraints)
    _no_collision(opt, v)


def _no_collision(opt: Any, v: SmtVariables) -> None:
    """Clause 4 of Definition 4.2.2: ``act^a_t != act^{a'}_t`` for ``a != a'``."""
    if v.A < 2 or v.T == 0:
        return
    if v.A >= 3:
        # Distinct's all-different propagator is cheaper than A*(A-1)/2
        # pairwise disequalities once A >= 3.
        from z3 import Distinct

        opt.add(*[Distinct([v.action[a][t] for a in range(v.A)]) for t in range(v.T)])
    else:
        opt.add(*[v.action[0][t] != v.action[1][t] for t in range(v.T)])


# ---------------------------------------------------------------------------
# Definition 4.2.3: protocol constraint [Prot]
# ---------------------------------------------------------------------------


def protocol_constraint(opt: Any, v: SmtVariables) -> None:
    """An agent claims only a sector that is unoccupied at that step."""
    from z3 import Implies

    constraints = [
        Implies(v.action[a][t] == i, v.owner[i][t] == -1)
        for t in range(v.T)
        for a in range(v.A)
        for i in range(v.S)
    ]
    if constraints:
        opt.add(*constraints)


# ---------------------------------------------------------------------------
# Definition 4.2.4: domain constraint [Dom]
# ---------------------------------------------------------------------------


def domain_constraint(opt: Any, v: SmtVariables) -> None:
    """Control values lie in ``Agt+`` and actions name sectors.

    In tool numbering: ``-1 <= owner <= A-1`` and ``0 <= action <= S-1``.
    """
    from z3 import And

    constraints = [
        And(v.owner[i][t] >= -1, v.owner[i][t] <= v.A - 1)
        for i in range(v.S)
        for t in range(v.T + 1)
    ]
    constraints += [
        And(v.action[a][t] >= 0, v.action[a][t] <= v.S - 1)
        for a in range(v.A)
        for t in range(v.T)
    ]
    if constraints:
        opt.add(*constraints)


# ---------------------------------------------------------------------------
# Definition 4.3.1: region witness constraint [Wit]
# ---------------------------------------------------------------------------


def witness_constraint(opt: Any, v: SmtVariables) -> None:
    """Clauses 1-6 of Definition 4.3.1, for every agent.

    1. ``in_{i,a} -> sec^i_m = a``;
    2. ``root_{i,a} -> in_{i,a}``;
    3. at most one root;
    4. ``0 <= lv_{i,a} <= max(m-1, 0)``;
    5. every non-root member has a neighbour in the witness with a smaller level;
    6. prefix form of the least-index root: ``not pre_1``,
       ``pre_{i+1} <-> (pre_i or in_i)`` and ``root_i -> not pre_i``.

    The adjacency is ``v.neighbors``, which is the block adjacency in the
    abstract game; levels then count blocks, and the bound ``max(m-1, 0)``
    uses the abstract horizon (Section 6.1).
    """
    from z3 import And, AtMost, BoolVal, Implies, Not, Or

    m = v.T
    S = v.S
    level_max = max(m - 1, 0)
    constraints: list[Any] = []
    for a in range(v.A):
        member, root, prefix, level = v.member[a], v.root[a], v.prefix[a], v.level[a]
        for i in range(S):
            constraints.append(Implies(member[i], v.owner[i][m] == a))  # 1
            constraints.append(Implies(root[i], member[i]))  # 2
            constraints.append(And(level[i] >= 0, level[i] <= level_max))  # 4
            nbs = sorted(v.neighbors.get(i, frozenset()))
            witness_nb = (
                Or([And(member[j], level[j] < level[i]) for j in nbs])
                if nbs
                else BoolVal(False)
            )
            constraints.append(Implies(And(member[i], Not(root[i])), witness_nb))  # 5
        if S:
            constraints.append(AtMost(*root, 1))  # 3
            constraints.append(Not(prefix[0]))  # 6
            for i in range(S - 1):
                constraints.append(prefix[i + 1] == Or(prefix[i], member[i]))
            for i in range(S):
                constraints.append(Implies(root[i], Not(prefix[i])))
    if constraints:
        opt.add(*constraints)


# ---------------------------------------------------------------------------
# Definition 4.3.2: reward constraint [Rew]
# ---------------------------------------------------------------------------


def reward_constraint(opt: Any, v: SmtVariables) -> None:
    """``reward_a`` is the (weighted) number of sectors of the witness region.

    With weights this is ``reward_a = sum_p omega(B_p) [in_{p,a}]``
    (Section 6.1).
    """
    from z3 import If, IntVal, Sum

    constraints = []
    for a in range(v.A):
        terms = [If(v.member[a][i], v.weight(i), 0) for i in range(v.S)]
        constraints.append(v.reward[a] == (Sum(terms) if terms else IntVal(0)))
    if constraints:
        opt.add(*constraints)


# ---------------------------------------------------------------------------
# Definition 4.3.3: qualitative objective constraint [phi_sat]
# ---------------------------------------------------------------------------


def qualitative_constraint(
    opt: Any, v: SmtVariables, demands: tuple[int, ...] | None = None
) -> None:
    """``reward_a >= d(a)`` for every agent; *demands* ``None`` means ``d = 0``."""
    if demands is None:
        demands = (0,) * v.A
    if len(demands) != v.A:
        raise ValueError("demands must have one entry per agent")
    if v.A:
        opt.add(*[v.reward[a] >= int(demands[a]) for a in range(v.A)])


# ---------------------------------------------------------------------------
# Definition 4.7.1: opponent constraint [Fix^a_pi]
# ---------------------------------------------------------------------------


def unoccupied_count_constraint(opt: Any, v: SmtVariables) -> None:
    """Define ``unocc^i_t``, the number of unoccupied sectors before *i* at step *t*.

    ``unocc^1_t = 0`` and ``unocc^{i+1}_t = unocc^i_t + [sec^i_t = 0]``
    (Definition 4.7.1).  The equalities only define the auxiliary
    variables, so adding them twice is harmless.
    """
    from z3 import If

    constraints = []
    for t in range(v.T):
        if v.S == 0:
            continue
        constraints.append(v.unocc[t][0] == 0)
        for i in range(v.S - 1):
            constraints.append(
                v.unocc[t][i + 1] == v.unocc[t][i] + If(v.owner[i][t] == -1, 1, 0)
            )
    if constraints:
        opt.add(*constraints)


def _state_at(v: SmtVariables, state: tuple[int, ...], t: int) -> Any:
    """``[s]_t``: the encoded state at step *t* equals *state*."""
    from z3 import And

    return And([v.owner[i][t] == state[i] for i in range(v.S)])


def _default_action_at(v: SmtVariables, agent: int, t: int) -> Any:
    """``act^b_t`` claims ``free_b`` of the state at step *t* (thesis agent ``b = agent + 1``)."""
    from z3 import And, Implies

    return And(
        [
            Implies(
                v.action[agent][t] == i,
                And(v.owner[i][t] == -1, v.unocc[t][i] == agent),
            )
            for i in range(v.S)
        ]
    )


def opponent_constraint(opt: Any, v: SmtVariables, play: Play, deviator: int) -> None:
    """``[Fix^a_pi]`` of Definition 4.7.1, with *deviator* as agent ``a``.

    For every opponent ``b != a`` and step ``t < m``: if the state at step
    ``t`` is ``s_t`` of *play*, ``b`` takes its recorded action
    ``ap_t(b)``; otherwise ``b`` claims the unoccupied sector with exactly
    ``b`` (tool index) unoccupied sectors before it, which is its default
    action.  Together with ``F`` this makes every opponent follow the
    canonical completion of *play* at every state that a run reaches
    (Lemma 4.7.1), also when the run returns to the play.
    """
    from z3 import Implies, Not

    if deviator < 0 or deviator >= v.A:
        raise ValueError("deviator index out of range")
    if play.num_agents != v.A or play.horizon != v.T:
        raise ValueError("the play must have the encoding's agents and horizon")
    if any(len(s) != v.S for s in play.states):
        raise ValueError("the play's states must have one entry per sector")
    if v.A == 1:
        return
    unoccupied_count_constraint(opt, v)
    constraints = []
    for t in range(v.T):
        on_play = _state_at(v, play.states[t], t)
        for b in range(v.A):
            if b == deviator:
                continue
            constraints.append(Implies(on_play, v.action[b][t] == play.actions[t][b]))
            constraints.append(Implies(Not(on_play), _default_action_at(v, b, t)))
    if constraints:
        opt.add(*constraints)


# ---------------------------------------------------------------------------
# Structural constraints (Q-IBIS, CEGAR-NE)
# ---------------------------------------------------------------------------


def weight_balance_constraint(opt: Any, v: SmtVariables, target_weight: int) -> None:
    """Each agent's total controlled weight equals *target_weight* at the final step.

    Optional constraint for the abstract RCMAS; it is not part of the
    encoding of Definition 6.1.2.  Uses Z3's ``PbEq`` (pseudo-boolean
    equality) when available, and ``Sum``/``If`` otherwise.
    """
    final_t = v.T
    assert v.weights is not None

    try:
        from z3 import PbEq

        opt.add(
            *[
                PbEq(
                    [(v.owner[s][final_t] == a, v.weights[s]) for s in range(v.S)],
                    target_weight,
                )
                for a in range(v.A)
            ]
        )
    except ImportError:  # pragma: no cover
        from z3 import If, Sum

        for a in range(v.A):
            total = Sum(
                [If(v.owner[s][final_t] == a, v.weights[s], 0) for s in range(v.S)]
            )
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


def _occupied(state: tuple[int, ...]) -> int:
    return sum(1 for o in state if o != -1)


def fixed_policy_constraint(
    opt: Any,
    v: SmtVariables,
    territory: Territory,
    fixed_policy_by_agent: tuple[Policy | None, ...],
    enforce_state_only_for_agents: tuple[int, ...] = (),
) -> None:
    """Constrain agents to follow state-based policies with the default fallback.

    An agent with a policy takes the mapped action at every mapped state that
    a run reaches, and its default action (Definition 4.4.2) at every other
    state; a ``None`` entry also means the default action.  For the policy of
    a play's recorded actions (``CanonicalCompletion.policy``) this is the
    canonical completion, as in :func:`opponent_constraint`.  Agents whose
    policy is ``None`` are unconstrained.

    A state with ``k`` occupied sectors can only occur at step ``k / |Agt|``
    (Lemma 4.4.1), so each mapped state guards only that step.
    *enforce_state_only_for_agents* is accepted for backward compatibility:
    the states of a run are pairwise distinct (Lemma 4.4.1(3)), so the
    same-state-same-action condition it asked for always holds.
    """
    from z3 import And, Implies, Not, Or

    S, A, T = v.S, v.A, v.T

    if len(fixed_policy_by_agent) != A:
        raise ValueError("fixed_policy_by_agent must have length == num_agents")
    for a in enforce_state_only_for_agents:
        if a < 0 or a >= A:
            raise ValueError(
                "enforce_state_only_for_agents contains invalid agent index"
            )

    mapped: list[dict[int, list[tuple[tuple[int, ...], int]]]] = []
    for a in range(A):
        policy = fixed_policy_by_agent[a]
        by_step: dict[int, list[tuple[tuple[int, ...], int]]] = {}
        if policy is not None:
            for state, chosen in policy.items():
                if len(state) != S:
                    raise ValueError(
                        "fixed_policy_by_agent has a state key with wrong length"
                    )
                if chosen is None:
                    continue
                sidx = territory.index_of(chosen)
                if sidx is None:
                    raise ValueError(
                        f"fixed policy action {chosen} is not in territory"
                    )
                occ = _occupied(state)
                if A > 0 and occ % A == 0 and occ // A < T:
                    by_step.setdefault(occ // A, []).append((tuple(state), sidx))
        mapped.append(by_step)

    if any(p is not None for p in fixed_policy_by_agent):
        unoccupied_count_constraint(opt, v)

    constraints = []
    for a in range(A):
        if fixed_policy_by_agent[a] is None:
            continue
        for t in range(T):
            guards = []
            for state, sidx in sorted(mapped[a].get(t, [])):
                on_state = _state_at(v, state, t)
                guards.append(on_state)
                constraints.append(Implies(on_state, v.action[a][t] == sidx))
            off = Not(Or(guards)) if guards else None
            default = _default_action_at(v, a, t)
            constraints.append(Implies(off, default) if off is not None else default)
    if constraints:
        opt.add(*constraints)


def action_candidates_constraint(
    opt: Any,
    v: SmtVariables,
    territory: Territory,
    action_candidates_by_agent: tuple[ActionCandidates | None, ...],
) -> None:
    """Restrict agent actions to candidate sets at given states (``[Cand^{a,i}]``, Section 4.9).

    ``[s]_t -> act^a_t in Cand(s)`` for every candidate state *s*; a state
    with ``k`` occupied sectors is guarded only at step ``k / |Agt|``, the
    only step at which a run can reach it.  ``None`` entries in a candidate
    set are ignored (agents cannot pass).
    """
    from z3 import Implies, Or

    S, A, T = v.S, v.A, v.T

    if len(action_candidates_by_agent) != A:
        raise ValueError("action_candidates_by_agent must have length == num_agents")

    constraints = []
    for a in range(A):
        cand = action_candidates_by_agent[a]
        if cand is None:
            continue
        for state in sorted(cand.keys()):
            if len(state) != S:
                raise ValueError(
                    "action_candidates_by_agent has a state key with wrong length"
                )
            allowed: set[int] = set()
            for c in cand[state]:
                if c is None:
                    continue
                sidx = territory.index_of(c)
                if sidx is None:
                    raise ValueError(f"candidate action {c} is not in territory")
                allowed.add(sidx)
            occ = _occupied(tuple(state))
            if A == 0 or occ % A != 0 or occ // A >= T:
                continue
            t = occ // A
            choice = (
                Or([v.action[a][t] == i for i in sorted(allowed)]) if allowed else False
            )
            constraints.append(Implies(_state_at(v, tuple(state), t), choice))
    if constraints:
        opt.add(*constraints)


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
            Not(
                And(
                    v.owner[i][final_t] == v.owner[j][final_t],
                    v.owner[i][final_t] != -1,
                )
            )
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

    consequent = (
        And(consequent_clauses)
        if len(consequent_clauses) > 1
        else consequent_clauses[0]
    )
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
                                        And(v.action[a][0] == sf, v.action[b][0] == ss),
                                        False,
                                    )
                                )
