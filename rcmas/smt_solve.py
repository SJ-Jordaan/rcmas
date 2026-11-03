"""Solver assembly for the encoding of Chapter 4.

:func:`build_base_encoding` builds the formula

    F = [Init] and [Evol] and [Prot] and [Dom] and [Wit] and [Rew] and [phi_sat]

once per run, as a list of Z3 expressions (Section 7.1), together with the
bound ``reward_a <= R_max`` that ``F`` implies
(:func:`~rcmas.smt_constraints.reward_bound_constraint`).  Every query of a
run replays this list into a fresh solver and adds its own constraints:

- :func:`find_feasible_from_base`: ``smt(F)``, the initial play of IBIS,
  S-IBIS and Q-IBIS (line 1 of Algorithm 1), optionally with the
  symmetry-breaking constraints;
- :func:`best_response_from_base`: the best-response query
  ``smt(G_a(pi), reward_a)`` with ``G_a(pi) = F and [Fix^a_pi]``
  (Definition 4.7.1), on Z3's optimisation engine;
- :func:`check_improvement_from_base`: ``smt(G_a(pi) and reward_a > c)``,
  the feasibility filter of S-IBIS and the query of VerifyNE;
- :func:`best_response_binsearch_from_base`: the feasibility filter and
  binary search of Algorithm 2 for one agent.

The symmetry-breaking constraints of Chapter 5 are never part of ``F``.
:func:`symmetry_constraints` adds them, and the tool uses it only for the
initial query of IBIS, S-IBIS and Q-IBIS and for satisfying and
collectively optimal synthesis (:func:`solve_smt_game`,
:func:`solve_collective_optimality`), never in a best-response or
verification query (Section 5.4.5 and Section 7.1).

Rewards.  The witness encoding is exact for lower bounds and maximisation
only (Remark 4.3.2), so :class:`SmtSolution` reports the rewards of
Definition 3.9.2 evaluated on the final state of the model's play
(``payoff_by_agent``), under the encoding's adjacency and weights, and the
model values of ``reward_a`` separately (``encoded_reward_by_agent``).

Solver answers.  Every query reports ``reason="unknown"`` when Z3 answers
unknown (including a per-query timeout); callers must stop on it and never
read it as unsatisfiable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, cast

from .model import Coord, State, Territory, rewards_from_owner
from .smt_constraints import (
    action_candidates_constraint,
    domain_constraint,
    evolution_constraint,
    fixed_actions_constraint,
    fixed_policy_constraint,
    init_constraint,
    max_reward,
    opponent_constraint,
    protocol_constraint,
    qualitative_constraint,
    reward_bound_constraint,
    reward_constraint,
    weight_balance_constraint,
    witness_constraint,
)
from .smt_objectives import collective_objective, individual_objective
from .smt_variables import SmtVariables, create_variables
from .strategy import ActionCandidates, Play, Policy

ExtraConstraints = Callable[[Any, SmtVariables], None]
"""A function that adds constraints over the variables to a solver."""


# ---------------------------------------------------------------------------
# Solution dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmtSolution:
    """Result of one SMT query.

    Attributes:
        is_sat: the query has a model.
        reason: ``"sat"``, ``"unsat"`` or ``"unknown"``.
        final_state: the final state ``s_m`` of the model's play.
        actions_by_round: the profiles of the play, as coordinates.
        owner_by_round: the control vectors ``s_0, ..., s_m``.
        payoff_by_agent: rewards of Definition 3.9.2 on ``s_m`` (weighted
            component sizes in the abstract game), never model values.
        encoded_reward_by_agent: model values of ``reward_a`` (at most the
            rewards, Lemma 4.3.1).
        best_seed_by_agent: root of each agent's witness region (debug only).
        best_region_by_agent: members of each witness region (debug only).
        play: the play of the model (Definition 4.4.1).
    """

    is_sat: bool
    reason: str
    final_state: State | None
    actions_by_round: tuple[tuple[Coord | None, ...], ...] | None
    owner_by_round: tuple[tuple[int, ...], ...] | None = None
    payoff_by_agent: tuple[int, ...] | None = None
    encoded_reward_by_agent: tuple[int, ...] | None = None
    best_seed_by_agent: tuple[int | None, ...] | None = None
    best_region_by_agent: tuple[tuple[Coord, ...] | None, ...] | None = None
    play: Play | None = None

    @property
    def reward_by_agent(self) -> tuple[int, ...] | None:
        """Rewards of Definition 3.9.2 on the final state (alias of ``payoff_by_agent``)."""
        return self.payoff_by_agent

    @property
    def is_unknown(self) -> bool:
        return self.reason == "unknown"


def _no_model(check_res: Any) -> SmtSolution:
    from z3 import unknown

    if check_res == unknown:
        return SmtSolution(False, "unknown", None, None)
    return SmtSolution(False, "unsat", None, None)


def _check(solver: Any, *assumptions: Any) -> Any:
    """Run ``solver.check``.  Kept as one function so tests can inject answers."""
    return solver.check(*assumptions)


# ---------------------------------------------------------------------------
# Constraint caching
# ---------------------------------------------------------------------------


class ConstraintCollector:
    """Accumulates Z3 constraint expressions for later replay into a solver.

    Duck-types as a solver for the constraint functions, which only call
    ``.add()``.
    """

    __slots__ = ("_constraints",)

    def __init__(self) -> None:
        self._constraints: list[Any] = []

    def add(self, *args: Any) -> None:
        self._constraints.extend(args)

    @property
    def constraints(self) -> list[Any]:
        return self._constraints


@dataclass(frozen=True, slots=True)
class BaseEncoding:
    """The formula ``F`` of a game, built once and replayed into every query.

    Attributes:
        variables: the Z3 variables.
        constraints: the expressions of ``F`` (and of the optional weight
            balance constraint).
        territory: the territory whose enumeration numbers the sectors.
        demands: the demands posted in ``[phi_sat]`` (``None`` = all 0).
    """

    variables: SmtVariables
    constraints: list[Any]
    territory: Territory
    demands: tuple[int, ...] | None = None

    @property
    def num_agents(self) -> int:
        return self.variables.A

    @property
    def horizon(self) -> int:
        return self.variables.T

    @property
    def max_reward(self) -> int:
        """``R_max`` of S-IBIS: ``m``, or the largest total weight of ``m`` sectors (Section 6.1)."""
        return max_reward(self.variables)

    def rewards_of(
        self, owner_by_index: tuple[int, ...] | list[int]
    ) -> tuple[int, ...]:
        """Rewards of Definition 3.9.2 (or 6.1.2(4)) of a final control vector."""
        v = self.variables
        return rewards_from_owner(owner_by_index, v.A, v.neighbors, v.weights)


def build_base_encoding(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    demands: tuple[int, ...] | None = None,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    symmetry_breaking: bool = False,
    require_victory: bool = False,
    max_region_size: int | None = None,
) -> BaseEncoding:
    """Build the formula ``F`` of Section 4.2 once for reuse across queries.

    *demands* gives ``d(a)`` for ``[phi_sat]`` (``None`` means every demand
    is 0).  *weights* and *custom_neighbors* encode the abstract RCMAS of
    Definition 6.1.2.  *weight_balance_target* adds the optional weight
    balance constraint of the abstract game.

    *symmetry_breaking*, *require_victory* and *max_region_size* are
    accepted for backward compatibility and have no effect.  Symmetry
    constraints never belong to ``F`` (use :func:`symmetry_constraints` on
    the initial query), the thesis has no victory condition (at the horizon
    ``m`` every agent controls ``m`` sectors, Lemma 3.10.1), and the
    witness encoding needs no pruning.
    """
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    if demands is not None and len(demands) != num_agents:
        raise ValueError("demands must have one entry per agent")
    if symmetry_breaking:
        warnings.warn(
            "build_base_encoding(symmetry_breaking=True) has no effect: symmetry "
            "constraints are added only to the initial query (symmetry_constraints)",
            DeprecationWarning,
            stacklevel=2,
        )
    _ = (require_victory, max_region_size)

    v = create_variables(
        territory,
        num_agents,
        horizon,
        weights=weights,
        custom_neighbors=custom_neighbors,
    )

    collector = ConstraintCollector()
    init_constraint(collector, v)
    evolution_constraint(collector, v)
    protocol_constraint(collector, v)
    domain_constraint(collector, v)
    witness_constraint(collector, v)
    reward_constraint(collector, v)
    qualitative_constraint(collector, v, demands)
    reward_bound_constraint(collector, v)

    if weight_balance_target is not None:
        weight_balance_constraint(collector, v, weight_balance_target)

    return BaseEncoding(
        variables=v,
        constraints=collector.constraints,
        territory=territory,
        demands=None if demands is None else tuple(demands),
    )


# ---------------------------------------------------------------------------
# Symmetry-breaking constraints (Chapter 5): initial queries only
# ---------------------------------------------------------------------------


def symmetry_constraints(
    opt: Any,
    v: SmtVariables,
    territory: Territory,
    demands: tuple[int, ...] | None = None,
) -> None:
    """Add the symmetry-breaking constraints of Chapter 5 to *opt*.

    Disconnected territories get the component-wise constraints; connected
    territories get the lex-leader and spatial canonicalisation
    constraints, and the conditional local constraints when the global
    symmetry group is trivial.  Use this only for satisfying and
    collectively optimal synthesis and for the query that produces the
    initial play of IBIS, S-IBIS and Q-IBIS (Section 5.4.5); best-response
    and verification queries are posed without it.

    The constraints are computed from the grid geometry of *territory*, so
    they do not apply to the synthetic territory of an abstract RCMAS.
    """
    from .smt_constraints import symmetry_plan_constraint
    from .symmetry import symmetry_breaking_plan

    if v.T == 0:
        return
    # One source for which constraints apply: the plan of rcmas.symmetry,
    # whose ``admits`` the oracle tests check against enumeration.
    symmetry_plan_constraint(opt, v, symmetry_breaking_plan(territory, v.A, demands))


# ---------------------------------------------------------------------------
# Queries over a cached formula
# ---------------------------------------------------------------------------


def _new_solver(kind: str, timeout_ms: int | None) -> Any:
    from z3 import Optimize, Solver

    solver = Optimize() if kind == "opt" else Solver()
    if timeout_ms is not None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        solver.set(timeout=timeout_ms)
    return solver


def _add_query_constraints(
    solver: Any,
    base: BaseEncoding,
    *,
    opponent_play: Play | None,
    deviator: int | None,
    fixed_policy_by_agent: tuple[Policy | None, ...] | None,
    enforce_state_only_for_agents: tuple[int, ...],
    action_candidates_by_agent: tuple[ActionCandidates | None, ...] | None,
    extra: ExtraConstraints | None,
) -> None:
    v = base.variables
    solver.add(*base.constraints)
    if opponent_play is not None:
        if deviator is None:
            raise ValueError("an opponent play needs the index of the deviating agent")
        opponent_constraint(solver, v, opponent_play, deviator)
    if fixed_policy_by_agent is not None:
        fixed_policy_constraint(
            solver,
            v,
            base.territory,
            fixed_policy_by_agent,
            enforce_state_only_for_agents,
        )
    if action_candidates_by_agent is not None:
        action_candidates_constraint(
            solver, v, base.territory, action_candidates_by_agent
        )
    if extra is not None:
        extra(solver, v)


def find_feasible_from_base(
    base: BaseEncoding,
    *,
    symmetry: bool = False,
    extra_constraints: ExtraConstraints | None = None,
    fixed_policy_by_agent: tuple[Policy | None, ...] | None = None,
    enforce_state_only_for_agents: tuple[int, ...] = (),
    debug: bool = False,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """``smt(F)``: any model of the formula (line 1 of Algorithms 1-3).

    With *symmetry* the symmetry-breaking constraints of the territory are
    added; *extra_constraints* adds further constraints to this query only
    (for example, the symmetry constraints of an abstract game).
    """
    v = base.variables
    solver = _new_solver("sat", timeout_ms)
    _add_query_constraints(
        solver,
        base,
        opponent_play=None,
        deviator=None,
        fixed_policy_by_agent=fixed_policy_by_agent,
        enforce_state_only_for_agents=enforce_state_only_for_agents,
        action_candidates_by_agent=None,
        extra=extra_constraints,
    )
    if symmetry:
        symmetry_constraints(solver, v, base.territory, base.demands)
    from z3 import sat

    res = _check(solver)
    if res != sat:
        return _no_model(res)
    return _extract_solution(solver.model(), base, debug)


def solve_from_base(
    base: BaseEncoding,
    *,
    objective: str | int = "sum",
    opponent_play: Play | None = None,
    fixed_policy_by_agent: tuple[Policy | None, ...] | None = None,
    action_candidates_by_agent: tuple[ActionCandidates | None, ...] | None = None,
    enforce_state_only_for_agents: tuple[int, ...] = (),
    extra_constraints: ExtraConstraints | None = None,
    debug: bool = False,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """Optimisation query over the cached formula.

    *objective* ``"sum"`` maximises ``f`` (Definition 4.6.1); an agent index
    ``a`` maximises ``reward_a``.  With *opponent_play* and an agent index,
    the query is the best-response query ``smt(G_a(pi), reward_a)`` of
    Definition 4.7.1, and *action_candidates_by_agent* restricts it as in
    Q-IBIS (the restricted query of Algorithm 3, line 14).
    """
    v = base.variables
    if objective != "sum" and not isinstance(objective, int):
        raise ValueError("objective must be 'sum' or an agent index")
    if isinstance(objective, int) and (objective < 0 or objective >= v.A):
        raise ValueError("objective agent index out of range")
    opt = _new_solver("opt", timeout_ms)
    _add_query_constraints(
        opt,
        base,
        opponent_play=opponent_play,
        deviator=objective if isinstance(objective, int) else None,
        fixed_policy_by_agent=fixed_policy_by_agent,
        enforce_state_only_for_agents=enforce_state_only_for_agents,
        action_candidates_by_agent=action_candidates_by_agent,
        extra=extra_constraints,
    )
    if objective == "sum":
        collective_objective(opt, v)
    else:
        individual_objective(opt, v, cast(int, objective))
    from z3 import sat

    res = _check(opt)
    if res != sat:
        return _no_model(res)
    return _extract_solution(opt.model(), base, debug)


def best_response_from_base(
    base: BaseEncoding,
    play: Play,
    agent: int,
    *,
    action_candidates_by_agent: tuple[ActionCandidates | None, ...] | None = None,
    debug: bool = False,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """The best-response query ``smt(G_a(pi), reward_a)`` of Definition 4.7.1.

    For a satisfying play the query always has a model (Lemma 4.7.2(3)), and
    the reward of its play is ``BR_a(pi)``; read it from
    ``payoff_by_agent[agent]``, which is evaluated on the final state.
    """
    return solve_from_base(
        base,
        objective=agent,
        opponent_play=play,
        action_candidates_by_agent=action_candidates_by_agent,
        debug=debug,
        timeout_ms=timeout_ms,
    )


def check_improvement_from_base(
    base: BaseEncoding,
    *,
    agent: int,
    threshold: int,
    opponent_play: Play | None = None,
    fixed_policy_by_agent: tuple[Policy | None, ...] | None = None,
    enforce_state_only_for_agents: tuple[int, ...] = (),
    debug: bool = False,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """``smt(G_a(pi) and reward_a > threshold)``: can *agent* exceed *threshold*?

    With *opponent_play* this is the feasibility filter of S-IBIS (line 11 of
    Algorithm 2) and the query that VerifyNE poses for each agent (Section
    6.3).  A model is a profitable admissible deviation when *threshold* is
    the agent's current reward (Lemma 4.7.2(1)).  The legacy
    *fixed_policy_by_agent* form constrains the opponents by policies.
    """
    v = base.variables
    if agent < 0 or agent >= v.A:
        raise ValueError("agent index out of range")
    solver = _new_solver("sat", timeout_ms)
    _add_query_constraints(
        solver,
        base,
        opponent_play=opponent_play,
        deviator=agent,
        fixed_policy_by_agent=fixed_policy_by_agent,
        enforce_state_only_for_agents=enforce_state_only_for_agents,
        action_candidates_by_agent=None,
        extra=None,
    )
    solver.add(v.reward[agent] > threshold)
    from z3 import sat

    res = _check(solver)
    if res != sat:
        return _no_model(res)
    return _extract_solution(solver.model(), base, debug)


@dataclass(frozen=True, slots=True)
class BinarySearchResult:
    """Outcome of the S-IBIS best-response computation for one agent.

    Attributes:
        reason: ``"improved"`` (``solution`` is ``beta_a``), ``"no_improvement"``
            (the feasibility filter is unsatisfiable, so ``IP = 0``) or
            ``"unknown"`` (Z3 answered unknown; the run must stop).
        solution: the last satisfiable model ``beta_a``, if any.
        checks: number of satisfiability checks posed.
    """

    reason: str
    solution: SmtSolution | None
    checks: int


def best_response_binsearch_from_base(
    base: BaseEncoding,
    play: Play,
    agent: int,
    *,
    current_reward: int,
    max_reward: int | None = None,
    debug: bool = False,
    timeout_ms: int | None = None,
) -> BinarySearchResult:
    """Lines 11-24 of Algorithm 2 for one agent.

    The feasibility filter ``smt(G_a(pi) and reward_a > r_a(pi))`` is
    followed, if satisfiable, by a binary search over ``[lo, hi]`` with
    ``lo`` the filter model's value of ``reward_a`` and ``hi = R_max``; each
    probe ``smt(G_a(pi) and reward_a >= mid)`` either raises ``lo`` to the
    probe model's value of ``reward_a`` (Remark 4.8.1) or lowers ``hi`` to
    ``mid - 1``.  By Theorem 4.8.1 the last model's play has reward
    ``BR_a(pi)``.  One incremental solver serves all checks.
    """
    from z3 import sat, unknown

    v = base.variables
    if agent < 0 or agent >= v.A:
        raise ValueError("agent index out of range")
    hi = base.max_reward if max_reward is None else max_reward
    solver = _new_solver("sat", timeout_ms)
    _add_query_constraints(
        solver,
        base,
        opponent_play=play,
        deviator=agent,
        fixed_policy_by_agent=None,
        enforce_state_only_for_agents=(),
        action_candidates_by_agent=None,
        extra=None,
    )

    checks = 1
    solver.push()
    solver.add(v.reward[agent] > current_reward)
    res = _check(solver)
    if res == unknown:
        return BinarySearchResult("unknown", None, checks)
    if res != sat:
        return BinarySearchResult("no_improvement", None, checks)
    best = _extract_solution(solver.model(), base, debug)
    solver.pop()
    assert best.encoded_reward_by_agent is not None
    lo = best.encoded_reward_by_agent[agent]

    while lo < hi:
        mid = (lo + hi + 1) // 2
        solver.push()
        solver.add(v.reward[agent] >= mid)
        res = _check(solver)
        checks += 1
        if res == unknown:
            return BinarySearchResult("unknown", best, checks)
        if res == sat:
            best = _extract_solution(solver.model(), base, debug)
            assert best.encoded_reward_by_agent is not None
            lo = best.encoded_reward_by_agent[agent]
        else:
            hi = mid - 1
        solver.pop()
    return BinarySearchResult("improved", best, checks)


# ---------------------------------------------------------------------------
# One-shot synthesis (Sections 4.5 and 4.6)
# ---------------------------------------------------------------------------


def solve_smt_game(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    objective: str | int | None = "sum",
    fixed_actions_by_round: tuple[tuple[Coord | None, ...], ...] | None = None,
    fixed_policy_by_agent: tuple[Policy | None, ...] | None = None,
    action_candidates_by_agent: tuple[ActionCandidates | None, ...] | None = None,
    enforce_state_only_for_agents: tuple[int, ...] = (),
    require_victory: bool = False,
    debug: bool = False,
    timeout_ms: int | None = None,
    symmetry_breaking: bool = False,
    demands: tuple[int, ...] | None = None,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
) -> SmtSolution:
    """One query over ``F`` with an optional objective.

    *objective* ``"sum"`` gives ``smt(F, f)`` (collectively optimal
    synthesis, Theorem 4.6.1), an agent index maximises that agent's
    reward, and ``None`` gives ``smt(F)`` (satisfying synthesis,
    Theorem 4.5.1).  *symmetry_breaking* adds the constraints of Chapter 5,
    which preserve satisfying and collectively optimal strategies (Section
    5.4.5); they are therefore added only for these two queries (objective
    ``None`` or ``"sum"`` with no fixed actions, policies or candidates), and
    any other query ignores the flag with a warning.  *require_victory* has
    no effect (see :func:`build_base_encoding`).
    """
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    if objective is not None and objective != "sum" and not isinstance(objective, int):
        raise ValueError("objective must be 'sum', an agent index or None")

    base = build_base_encoding(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        demands=demands,
        weights=weights,
        custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target,
        require_victory=require_victory,
    )

    synthesis_query = (
        objective in (None, "sum")
        and fixed_actions_by_round is None
        and fixed_policy_by_agent is None
        and action_candidates_by_agent is None
    )
    use_symmetry = symmetry_breaking and synthesis_query
    if symmetry_breaking and not synthesis_query:
        warnings.warn(
            "symmetry_breaking ignored: the constraints of Chapter 5 apply only to "
            "satisfying and collectively optimal synthesis",
            stacklevel=2,
        )

    def extra(opt: Any, v: SmtVariables) -> None:
        if fixed_actions_by_round is not None:
            fixed_actions_constraint(opt, v, territory, fixed_actions_by_round)
        if use_symmetry:
            symmetry_constraints(opt, v, territory, demands)

    if objective is None:
        if action_candidates_by_agent is not None:
            raise ValueError("candidate constraints need an objective")
        return find_feasible_from_base(
            base,
            extra_constraints=extra,
            fixed_policy_by_agent=fixed_policy_by_agent,
            enforce_state_only_for_agents=enforce_state_only_for_agents,
            debug=debug,
            timeout_ms=timeout_ms,
        )
    return solve_from_base(
        base,
        objective=objective,
        fixed_policy_by_agent=fixed_policy_by_agent,
        action_candidates_by_agent=action_candidates_by_agent,
        enforce_state_only_for_agents=enforce_state_only_for_agents,
        extra_constraints=extra,
        debug=debug,
        timeout_ms=timeout_ms,
    )


def solve_collective_optimality(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    debug: bool = False,
    symmetry_breaking: bool = False,
    demands: tuple[int, ...] | None = None,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """``smt(F, f)``: a collectively optimal play (Theorem 4.6.1)."""
    return solve_smt_game(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        objective="sum",
        debug=debug,
        timeout_ms=timeout_ms,
        symmetry_breaking=symmetry_breaking,
        demands=demands,
    )


def solve_satisfying(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    debug: bool = False,
    symmetry_breaking: bool = False,
    demands: tuple[int, ...] | None = None,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """``smt(F)``: a satisfying play (Theorem 4.5.1)."""
    return solve_smt_game(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        objective=None,
        debug=debug,
        timeout_ms=timeout_ms,
        symmetry_breaking=symmetry_breaking,
        demands=demands,
    )


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------


def _extract_solution(
    model: Any, base: BaseEncoding, debug: bool = False
) -> SmtSolution:
    """Read the play of a model (Definition 4.4.1) and evaluate its rewards."""
    from z3 import is_true

    v = base.variables
    territory = base.territory
    S, A, T = v.S, v.A, v.T
    sectors = v.sectors

    def val(term: Any) -> int:
        ref = cast(Any, model.eval(term, model_completion=True))
        return int(ref.as_long())

    actions_idx = tuple(tuple(val(v.action[a][t]) for a in range(A)) for t in range(T))
    owner_by_round = tuple(
        tuple(val(v.owner[i][t]) for i in range(S)) for t in range(T + 1)
    )
    actions_by_round = tuple(
        tuple(None if i < 0 else sectors[i] for i in step) for step in actions_idx
    )
    final_owner = owner_by_round[T]
    final_state = State(
        territory=territory, num_agents=A, owner_by_index=final_owner, round_index=T
    )
    play = Play(owner_by_round, actions_idx, A)
    rewards = base.rewards_of(final_owner)
    encoded = tuple(val(v.reward[a]) for a in range(A))

    seeds: tuple[int | None, ...] | None = None
    regions: tuple[tuple[Coord, ...] | None, ...] | None = None
    if debug:
        seed_list: list[int | None] = []
        region_list: list[tuple[Coord, ...] | None] = []
        for a in range(A):
            members = [
                i
                for i in range(S)
                if is_true(model.eval(v.member[a][i], model_completion=True))
            ]
            roots = [
                i
                for i in range(S)
                if is_true(model.eval(v.root[a][i], model_completion=True))
            ]
            seed_list.append(roots[0] if roots else None)
            region_list.append(tuple(sectors[i] for i in members) if members else None)
        seeds = tuple(seed_list)
        regions = tuple(region_list)

    return SmtSolution(
        True,
        "sat",
        final_state,
        actions_by_round,
        owner_by_round=owner_by_round,
        payoff_by_agent=rewards,
        encoded_reward_by_agent=encoded,
        best_seed_by_agent=seeds,
        best_region_by_agent=regions,
        play=play,
    )
