"""SMT solver assembly: compose constraints + objectives, call Z3, extract solution.

This module provides :func:`solve_smt_game`, the main entry point for SMT-based
synthesis.  It composes the constraint functions from :mod:`smt_constraints` and
the objectives from :mod:`smt_objectives`, invokes Z3, and extracts a
:class:`SmtSolution` from the model.

For iterative algorithms (IBIS, Q-IBIS), :func:`build_base_encoding` pre-builds
the expensive constraint expressions once, and :func:`solve_from_base` replays
them into fresh solver instances per query—avoiding redundant O(S³) expression
construction on each best-response call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from .model import Coord, State, Territory
from .smt_constraints import (
    action_candidates_constraint,
    adjacency_constraint,
    cohesive_region_constraint,
    collision_constraint,
    evolution_constraint,
    fixed_actions_constraint,
    fixed_policy_constraint,
    init_constraint,
    protocol_constraint,
    reward_constraint,
    size_constraint,
    symmetry_breaking_constraint,
    victory_constraint,
    weight_balance_constraint,
)
from .smt_objectives import qualitative_objective, quantitative_objective
from .smt_variables import SmtVariables, create_variables
from .strategy import ActionCandidates, Policy


# ---------------------------------------------------------------------------
# Solution dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SmtSolution:
    is_sat: bool
    reason: str
    final_state: State | None
    actions_by_round: tuple[tuple[Coord | None, ...], ...] | None
    # Debug/inspection data (only populated when debug=True)
    owner_by_round: tuple[tuple[int, ...], ...] | None = None
    payoff_by_agent: tuple[int, ...] | None = None
    size_by_seed: tuple[tuple[int, ...], ...] | None = None
    best_seed_by_agent: tuple[int | None, ...] | None = None
    best_region_by_agent: tuple[tuple[Coord, ...] | None, ...] | None = None


# ---------------------------------------------------------------------------
# Incremental solving: constraint caching for iterative algorithms
# ---------------------------------------------------------------------------

class ConstraintCollector:
    """Accumulates Z3 constraint expressions for later replay into a solver.

    Duck-types as an ``Optimize`` for the purpose of constraint functions
    (which only call ``.add()``).
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
    """Pre-built base constraints and variables, reusable across solver calls.

    The Z3 expression trees are Python objects that can be added to any
    ``Optimize`` instance that shares the same variable references.
    """

    variables: SmtVariables
    constraints: list[Any]
    territory: Territory


def build_base_encoding(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    require_victory: bool = False,
    symmetry_breaking: bool = False,
    demands: tuple[int, ...] | None = None,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
) -> BaseEncoding:
    """Build variables and base constraints once for reuse across solver calls.

    This is the expensive step—especially the O(S³) cohesive-region constraint.
    The returned :class:`BaseEncoding` can be passed to :func:`solve_from_base`
    for each per-agent best-response query without rebuilding expressions.
    """
    v = create_variables(
        territory, num_agents, horizon,
        weights=weights, custom_neighbors=custom_neighbors,
    )

    collector = ConstraintCollector()

    # Def 16-23: core constraints
    init_constraint(collector, v)
    protocol_constraint(collector, v)
    collision_constraint(collector, v)
    evolution_constraint(collector, v)
    adjacency_constraint(collector, v)
    cohesive_region_constraint(collector, v)
    size_constraint(collector, v)
    reward_constraint(collector, v)

    # Optional structural constraints (stable across queries)
    if require_victory:
        victory_constraint(collector, v)

    if weight_balance_target is not None:
        weight_balance_constraint(collector, v, weight_balance_target)

    if symmetry_breaking:
        from .symmetry import symmetry_info
        sym = symmetry_info(territory)
        symmetry_breaking_constraint(collector, v, sym, demands=demands)

    return BaseEncoding(
        variables=v,
        constraints=collector.constraints,
        territory=territory,
    )


def solve_from_base(
    base: BaseEncoding,
    *,
    objective: str | int = "sum",
    fixed_policy_by_agent: tuple[Policy | None, ...] | None = None,
    action_candidates_by_agent: tuple[ActionCandidates | None, ...] | None = None,
    enforce_state_only_for_agents: tuple[int, ...] = (),
    debug: bool = False,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """Solve using pre-built base constraints (avoids rebuilding expressions).

    Creates a fresh ``Optimize`` instance, replays the cached base constraints,
    adds per-query constraints (fixed policies, candidates, objective), and
    solves.
    """
    try:
        from z3 import Optimize, is_true, sat, unknown
    except Exception as e:  # pragma: no cover
        raise RuntimeError("z3-solver is required for SMT solving") from e

    v = base.variables
    territory = base.territory

    opt = Optimize()
    if timeout_ms is not None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        opt.set(timeout=timeout_ms)

    # Replay cached base constraints
    for c in base.constraints:
        opt.add(c)

    # Per-query constraints
    if fixed_policy_by_agent is not None:
        fixed_policy_constraint(opt, v, territory, fixed_policy_by_agent, enforce_state_only_for_agents)

    if action_candidates_by_agent is not None:
        action_candidates_constraint(opt, v, territory, action_candidates_by_agent)

    # Objective (Def 24/25)
    if objective == "sum":
        qualitative_objective(opt, v)
    elif isinstance(objective, int):
        quantitative_objective(opt, v, objective)
    else:
        raise ValueError("objective must be 'sum' or an agent index")

    # --- Solve ---
    check_res = opt.check()
    if check_res != sat:
        if check_res == unknown:
            return SmtSolution(False, "unknown", None, None)
        return SmtSolution(False, "unsat", None, None)

    model = opt.model()
    return _extract_solution(model, v, territory, v.A, v.T, debug, is_true)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def solve_collective_optimality(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    debug: bool = False,
    symmetry_breaking: bool = False,
    demands: tuple[int, ...] | None = None,
) -> SmtSolution:
    """One-shot collective-optimality solve (maximise total payoff)."""
    return solve_smt_game(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        objective="sum",
        require_victory=False,
        debug=debug,
        symmetry_breaking=symmetry_breaking,
        demands=demands,
    )


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_smt_game(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    objective: str | int = "sum",
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
    """Generic SMT solve for RCMAS game dynamics.

    Assembles all constraints from Def 16-23, adds the requested objective
    (Def 24 or 25), and returns the extracted solution.
    """
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")

    try:
        from z3 import Optimize, is_true, sat, unknown
    except Exception as e:  # pragma: no cover
        raise RuntimeError("z3-solver is required for SMT solving") from e

    # --- Build solver ---
    opt = Optimize()
    if timeout_ms is not None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        opt.set(timeout=timeout_ms)

    v = create_variables(
        territory, num_agents, horizon,
        weights=weights, custom_neighbors=custom_neighbors,
    )

    # Def 16-23: core constraints
    init_constraint(opt, v)
    protocol_constraint(opt, v)
    collision_constraint(opt, v)
    evolution_constraint(opt, v)
    adjacency_constraint(opt, v)
    cohesive_region_constraint(opt, v)
    size_constraint(opt, v)
    reward_constraint(opt, v)

    # Optional structural constraints
    if require_victory:
        victory_constraint(opt, v)

    if weight_balance_target is not None:
        weight_balance_constraint(opt, v, weight_balance_target)

    if fixed_actions_by_round is not None:
        fixed_actions_constraint(opt, v, territory, fixed_actions_by_round)

    if fixed_policy_by_agent is not None:
        fixed_policy_constraint(opt, v, territory, fixed_policy_by_agent, enforce_state_only_for_agents)

    if action_candidates_by_agent is not None:
        action_candidates_constraint(opt, v, territory, action_candidates_by_agent)

    if symmetry_breaking:
        from .symmetry import symmetry_info
        sym = symmetry_info(territory)
        symmetry_breaking_constraint(opt, v, sym, demands=demands)

    # Objective (Def 24/25)
    if objective == "sum":
        qualitative_objective(opt, v)
    elif isinstance(objective, int):
        quantitative_objective(opt, v, objective)
    else:
        raise ValueError("objective must be 'sum' or an agent index")

    # --- Solve ---
    check_res = opt.check()
    if check_res != sat:
        if check_res == unknown:
            return SmtSolution(False, "unknown", None, None)
        return SmtSolution(False, "unsat", None, None)

    model = opt.model()

    # --- Extract solution ---
    return _extract_solution(model, v, territory, num_agents, horizon, debug, is_true)


def _extract_solution(
    model: Any,
    v: SmtVariables,
    territory: Territory,
    num_agents: int,
    horizon: int,
    debug: bool,
    is_true: Any,
) -> SmtSolution:
    """Extract an SmtSolution from a Z3 model."""
    S, A, T = v.S, v.A, v.T
    sectors = v.sectors
    final_t = T

    # Actions per round
    actions_by_round: list[tuple[Coord | None, ...]] = []
    for t in range(T):
        step: list[Coord | None] = []
        for a in range(A):
            chosen_ref = cast(Any, model.eval(v.action[a][t], model_completion=True))
            chosen = int(chosen_ref.as_long())
            step.append(None if chosen < 0 else sectors[chosen])
        actions_by_round.append(tuple(step))

    # Final ownership
    final_owner_by_index: list[int] = []
    for s in range(S):
        own_ref = cast(Any, model.eval(v.owner[s][final_t], model_completion=True))
        final_owner_by_index.append(int(own_ref.as_long()))

    final_state = State(
        territory=territory,
        num_agents=A,
        owner_by_index=tuple(final_owner_by_index),
        round_index=T,
    )

    if not debug:
        return SmtSolution(True, "sat", final_state, tuple(actions_by_round))

    # --- Debug extraction ---
    owner_by_round: list[tuple[int, ...]] = []
    for t in range(T + 1):
        snapshot: list[int] = []
        for s in range(S):
            snap_ref = cast(Any, model.eval(v.owner[s][t], model_completion=True))
            snapshot.append(int(snap_ref.as_long()))
        owner_by_round.append(tuple(snapshot))

    payoff_by_agent: list[int] = []
    for a in range(A):
        p_ref = cast(Any, model.eval(v.payoff[a], model_completion=True))
        payoff_by_agent.append(int(p_ref.as_long()))

    size_by_seed: list[tuple[int, ...]] = []
    for i in range(S):
        row: list[int] = []
        for a in range(A):
            s_ref = cast(Any, model.eval(v.size[i][a], model_completion=True))
            row.append(int(s_ref.as_long()))
        size_by_seed.append(tuple(row))

    # Best seed and region witness
    def cr_true(i: int, j: int, a: int) -> bool:
        if i == j:
            return True
        lo, hi = (i, j) if i < j else (j, i)
        val = v.cr[(lo, hi, a)]
        b_ref = cast(Any, model.eval(val, model_completion=True))
        return is_true(b_ref)

    best_seed_by_agent: list[int | None] = []
    best_region_by_agent: list[tuple[Coord, ...] | None] = []

    for a in range(A):
        best_seed: int | None = None
        best_size = -1
        for i in range(S):
            val = size_by_seed[i][a]
            if val > best_size:
                best_size = val
                best_seed = i
        best_seed_by_agent.append(best_seed)

        if best_seed is None or best_size <= 0:
            best_region_by_agent.append(None)
            continue

        members: list[Coord] = []
        for j in range(S):
            if cr_true(best_seed, j, a):
                members.append(sectors[j])
        best_region_by_agent.append(tuple(members))

    return SmtSolution(
        True,
        "sat",
        final_state,
        tuple(actions_by_round),
        owner_by_round=tuple(owner_by_round),
        payoff_by_agent=tuple(payoff_by_agent),
        size_by_seed=tuple(size_by_seed),
        best_seed_by_agent=tuple(best_seed_by_agent),
        best_region_by_agent=tuple(best_region_by_agent),
    )
