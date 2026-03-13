"""IBIS and S-IBIS: Improvement-Based Iterative Synthesis.

IBIS (Algorithm 1, NFM2026) uses Z3 Optimize for best-response computation.
S-IBIS replaces Optimize with binary search over satisfiability checks,
which is typically much faster on large grids.

Both share the same iterative improvement loop via :func:`_ibis_core`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

from .model import Coord, Territory
from .smt_solve import (
    BaseEncoding,
    SmtSolution,
    build_base_encoding,
    find_feasible_from_base,
)
from .strategy import Policy, StateKey, freeze_profile, policy_from_solution


# ---------------------------------------------------------------------------
# Result type (shared by IBIS and S-IBIS)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IbisResult:
    is_sat: bool
    reason: str
    found_ne: bool
    iterations: int
    strategy: tuple[tuple[Coord | None, ...], ...] | None
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None


# Type alias for pluggable best-response functions.
BestResponseResult = tuple[int | None, float, int, Policy | None, int]
BestResponseFn = Callable[..., BestResponseResult]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def solve_ibis(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    seed: int = 0,
    max_iters: int = 25,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    demands: tuple[int, ...] | None = None,
) -> IbisResult:
    """Run IBIS: iterative best-response NE search via SMT (Alg 1).

    Uses Z3 Optimize for best-response computation.
    """
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if max_iters <= 0:
        raise ValueError("max_iters must be >= 1")

    return _ibis_core(
        territory=territory, num_agents=num_agents, horizon=horizon,
        seed=seed, max_iters=max_iters, progress=progress, timing=timing,
        timeout_ms=timeout_ms, symmetry=symmetry,
        weights=weights, custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target, demands=demands,
        find_best_improvement_fn=_find_best_improvement,
    )


def solve_sibis(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    seed: int = 0,
    max_iters: int = 25,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    demands: tuple[int, ...] | None = None,
) -> IbisResult:
    """Run S-IBIS: SAT-based iterative best-response NE search.

    Same algorithm structure as IBIS (Alg 1) but replaces Optimize-based
    best-response with binary search over satisfiability checks (Solver),
    which is typically much faster for large grids.
    """
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if max_iters <= 0:
        raise ValueError("max_iters must be >= 1")

    # Upper bound for binary search: max payoff any single agent can achieve.
    if weights is not None:
        max_payoff = sum(weights)
    else:
        max_payoff = len(territory) // num_agents

    br_fn = partial(_find_best_improvement_binsearch, max_payoff=max_payoff)

    return _ibis_core(
        territory=territory, num_agents=num_agents, horizon=horizon,
        seed=seed, max_iters=max_iters, progress=progress, timing=timing,
        timeout_ms=timeout_ms, symmetry=symmetry,
        weights=weights, custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target, demands=demands,
        find_best_improvement_fn=br_fn,
    )


# ---------------------------------------------------------------------------
# Implies-only policy model (replaces deterministic fallback everywhere)
# ---------------------------------------------------------------------------

def _add_implies_policy_for_agent(opt, v, territory: Territory, policy: Policy, agent: int) -> None:
    """Add Implies-only constraints for one agent's mapped-state policy.

    For each mapped state, adds ``Implies(state_match, action == chosen)``
    at every time step.  Unmapped states are unconstrained — no
    deterministic fallback.  Also adds state-only consistency (same
    state → same action) for the agent.
    """
    from z3 import And, Implies

    S, T = v.S, v.T

    if policy:
        for state, chosen_coord in policy.items():
            if chosen_coord is None:
                chosen_idx = -1
            else:
                sidx = territory.index_of(chosen_coord)
                if sidx is None:
                    continue
                chosen_idx = sidx

            for t in range(T):
                match_state = And([v.owner[s][t] == state[s] for s in range(S)])
                opt.add(Implies(match_state, v.action[agent][t] == chosen_idx))

    # State-only consistency.
    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            same_state = And([v.owner[s][t1] == v.owner[s][t2] for s in range(S)])
            opt.add(Implies(same_state, v.action[agent][t1] == v.action[agent][t2]))


def _add_payoff_floor_constraints(
    opt,
    v,
    territory: Territory,
    policies: list[Policy],
    deviating_agent: int,
    payoff_floors: tuple[int, ...],
) -> None:
    """Constrain opponents with payoff-floor model for the BR deviation check.

    Opponents follow their learned policy in mapped states (Implies only),
    are unconstrained in unmapped states, and must maintain at least their
    current payoff (preventing cooperation).  The deviating agent has
    state-only consistency but is otherwise free.
    """
    A = v.A

    for opp in range(A):
        if opp == deviating_agent:
            continue
        _add_implies_policy_for_agent(opt, v, territory, policies[opp], opp)
        opt.add(v.payoff[opp] >= payoff_floors[opp])

    # Deviating agent: state-only consistency only (free to deviate from policy).
    from z3 import And, Implies
    S, T = v.S, v.T
    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            same_state = And([v.owner[s][t1] == v.owner[s][t2] for s in range(S)])
            opt.add(Implies(same_state, v.action[deviating_agent][t1] == v.action[deviating_agent][t2]))


def _eval_joint_strategy(
    base_enc: BaseEncoding,
    policies: list[Policy],
    territory: Territory,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """Evaluate the current joint strategy under the Implies-only model.

    All agents follow their learned policies in mapped states, are
    unconstrained in unmapped states, and have state-only consistency.
    The objective maximizes the sum of all agents' payoffs (cooperative
    outcome consistent with current policies).
    """
    from .smt_solve import _extract_solution

    try:
        from z3 import Optimize, Sum, is_true, sat
    except Exception as e:  # pragma: no cover
        raise RuntimeError("z3-solver is required") from e

    v = base_enc.variables

    opt = Optimize()
    if timeout_ms is not None:
        opt.set(timeout=timeout_ms)
    opt.add(*base_enc.constraints)

    for a in range(v.A):
        _add_implies_policy_for_agent(opt, v, territory, policies[a], a)

    opt.maximize(Sum([v.payoff[a] for a in range(v.A)]))

    check_res = opt.check()
    if check_res != sat:
        return SmtSolution(False, "unsat", None, None)

    model = opt.model()
    return _extract_solution(model, v, territory, v.A, v.T, True, is_true)


# ---------------------------------------------------------------------------
# Core loop (shared by IBIS and S-IBIS)
# ---------------------------------------------------------------------------

def _ibis_core(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    seed: int = 0,
    max_iters: int = 25,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    demands: tuple[int, ...] | None = None,
    find_best_improvement_fn: BestResponseFn,
) -> IbisResult:
    """Iterative best-response loop with pluggable BR strategy."""

    def _log(msg: str) -> None:
        if progress:
            print(msg, file=sys.stderr, flush=True)

    _ = seed  # reserved for future perturbations

    policies: list[Policy] = [{} for _ in range(num_agents)]
    seen: set[tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]] = set()
    t0 = time.perf_counter()

    # Build base encoding once — caches expensive constraint expressions (O(S³))
    # For concrete games (no weights), prune cr variables for sector pairs
    # whose BFS distance exceeds the demand (max_region_size = S // A).
    mrs = None if weights is not None else len(territory) // num_agents
    base_enc = build_base_encoding(
        territory=territory, num_agents=num_agents, horizon=horizon,
        require_victory=True, symmetry_breaking=symmetry,
        weights=weights, custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target,
        demands=demands,
        max_region_size=mrs,
    )

    for it in range(1, max_iters + 1):
        iter_t0 = time.perf_counter()
        profile_key = freeze_profile(policies)
        if profile_key in seen:
            _log(f"iter={it} cycle_detected")
            base = _eval_joint_strategy(base_enc, policies, territory, timeout_ms)
            if timing:
                _log(f"iter={it} final_eval_time_s={time.perf_counter() - iter_t0:.3f}")
            return IbisResult(base.is_sat, "cycle", False, it - 1, base.actions_by_round, base.payoff_by_agent, base)
        seen.add(profile_key)

        base_t0 = time.perf_counter()
        all_empty = all(not p for p in policies)
        if all_empty:
            # First iteration: no policies fixed — use Solver (SAT) instead of
            # Optimize to find ANY feasible strategy quickly.  IBIS will
            # iteratively improve from whatever starting point we get.
            _log("using feasible-only solver (no optimisation)")
            base = find_feasible_from_base(
                base_enc,
                fixed_policy_by_agent=tuple(None for _ in policies),
                debug=True, timeout_ms=timeout_ms,
            )
        else:
            # Implies-only model: agents follow learned policies in mapped
            # states, are unconstrained in unmapped states.  Maximize sum
            # to get the cooperative baseline payoffs.
            base = _eval_joint_strategy(base_enc, policies, territory, timeout_ms)
        base_dt = time.perf_counter() - base_t0
        if not base.is_sat:
            return IbisResult(False, base.reason, False, it - 1, base.actions_by_round, None, base)
        if base.payoff_by_agent is None:
            return IbisResult(False, "missing_debug_payoff", False, it - 1, base.actions_by_round, None, base)

        base_payoff = base.payoff_by_agent
        if timing:
            _log(f"iter={it} base_payoff={base_payoff} base_solve_time_s={base_dt:.3f}")
        else:
            _log(f"iter={it} base_payoff={base_payoff}")

        if base.owner_by_round is not None and base.actions_by_round is not None:
            for a in range(num_agents):
                if not policies[a]:
                    policies[a] = policy_from_solution(
                        base.owner_by_round, base.actions_by_round, a,
                    )

        best_agent, best_ratio, best_delta, best_learned, best_learned_states = find_best_improvement_fn(
            territory=territory, num_agents=num_agents, horizon=horizon,
            policies=policies, base_payoff=base_payoff, timeout_ms=timeout_ms,
            it=it, _log=_log, timing=timing, symmetry=symmetry,
            weights=weights, custom_neighbors=custom_neighbors, weight_balance_target=weight_balance_target,
            demands=demands,
            base_encoding=base_enc,
        )

        if best_agent is None or best_learned is None:
            # No agent can improve under the payoff-floor opponent model.
            # Convergence under this model implies a verified NE — no
            # separate verification step is needed.
            if timing:
                _log(f"iter={it} converged_time_s={time.perf_counter() - iter_t0:.3f}")
                _log(f"done reason=ne iterations={it} total_time_s={time.perf_counter() - t0:.3f}")
            return IbisResult(True, "ne", True, it, base.actions_by_round, base_payoff, base)

        updated = dict(policies[best_agent])
        updated.update(best_learned)
        policies = [dict(p) for p in policies]
        policies[best_agent] = updated

        _log(f"iter={it} picked_agent={best_agent} picked_delta={best_delta} picked_ratio={best_ratio:.3f} picked_learned_states={best_learned_states}")
        if timing:
            _log(f"iter={it} updated_profile_states={[len(p) for p in policies]} iter_time_s={time.perf_counter() - iter_t0:.3f}")
        else:
            _log(f"iter={it} updated_profile_states={[len(p) for p in policies]}")

    final = _eval_joint_strategy(base_enc, policies, territory, timeout_ms)
    if timing:
        _log(f"done reason=max_iters iterations={max_iters} total_time_s={time.perf_counter() - t0:.3f}")

    return IbisResult(final.is_sat, "max_iters", False, max_iters, final.actions_by_round, final.payoff_by_agent, final)


# ---------------------------------------------------------------------------
# Best-response: Optimize (IBIS)
# ---------------------------------------------------------------------------

def _find_best_improvement(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    policies: list[Policy],
    base_payoff: tuple[int, ...],
    timeout_ms: int | None,
    it: int,
    _log,
    timing: bool,
    symmetry: bool = False,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    demands: tuple[int, ...] | None = None,
    base_encoding: BaseEncoding | None = None,
) -> tuple[int | None, float, int, Policy | None, int]:
    """Find the single agent with the best unilateral improvement.

    Uses Z3 Optimize to find each agent's optimal best-response under the
    payoff-floor opponent model: opponents follow learned policies in
    mapped states, are unconstrained in unmapped states, and must maintain
    at least their current payoff (preventing cooperation).
    """
    if base_encoding is None:
        raise ValueError("IBIS requires base_encoding")

    from .smt_solve import _extract_solution

    try:
        from z3 import Optimize, is_true, sat
    except Exception as e:  # pragma: no cover
        raise RuntimeError("z3-solver is required") from e

    v = base_encoding.variables

    best_agent: int | None = None
    best_ratio: float = 1.0
    best_delta: int = 0
    best_learned: Policy | None = None
    best_learned_states: int = 0

    for a in range(num_agents):
        opt = Optimize()
        if timeout_ms is not None:
            opt.set(timeout=timeout_ms)
        opt.add(*base_encoding.constraints)
        _add_payoff_floor_constraints(opt, v, territory, policies, a, base_payoff)
        opt.maximize(v.payoff[a])

        br_t0 = time.perf_counter()
        check_res = opt.check()
        br_dt = time.perf_counter() - br_t0

        if check_res != sat:
            if timing:
                _log(f"iter={it} agent={a} br=unsat time_s={br_dt:.3f}")
            else:
                _log(f"iter={it} agent={a} br=unsat")
            continue

        model = opt.model()
        br = _extract_solution(model, v, territory, v.A, v.T, True, is_true)
        if br.actions_by_round is None or br.payoff_by_agent is None:
            if timing:
                _log(f"iter={it} agent={a} br=unsat time_s={br_dt:.3f}")
            else:
                _log(f"iter={it} agent={a} br=unsat")
            continue

        new_payoff = br.payoff_by_agent[a]
        if timing:
            _log(f"iter={it} agent={a} br_payoff={new_payoff} time_s={br_dt:.3f}")
        else:
            _log(f"iter={it} agent={a} br_payoff={new_payoff}")

        if new_payoff > base_payoff[a]:
            assert br.owner_by_round is not None
            learned = policy_from_solution(br.owner_by_round, br.actions_by_round, a)
            delta = int(new_payoff - base_payoff[a])
            ratio = float(new_payoff) / float(base_payoff[a])
            learned_states = len(learned)

            is_better = (
                best_agent is None
                or ratio > best_ratio
                or (ratio == best_ratio and delta > best_delta)
                or (ratio == best_ratio and delta == best_delta and learned_states < best_learned_states)
            )
            if is_better:
                best_agent, best_ratio, best_delta, best_learned, best_learned_states = (
                    a, ratio, delta, learned, learned_states
                )
            _log(f"iter={it} agent={a} improved delta={delta} ratio={ratio:.3f} learned_states={learned_states}")

    return best_agent, best_ratio, best_delta, best_learned, best_learned_states


# ---------------------------------------------------------------------------
# Best-response: Binary search over SAT (S-IBIS)
# ---------------------------------------------------------------------------

def _find_best_improvement_binsearch(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    policies: list[Policy],
    base_payoff: tuple[int, ...],
    timeout_ms: int | None,
    it: int,
    _log,
    timing: bool,
    symmetry: bool = False,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    demands: tuple[int, ...] | None = None,
    base_encoding: BaseEncoding | None = None,
    max_payoff: int = 0,
) -> tuple[int | None, float, int, Policy | None, int]:
    """Find the single agent with the best unilateral improvement.

    Uses **incremental** binary search over Z3 Solver with push/pop instead
    of Z3 Optimize.  For each agent, one Solver is built with base constraints
    + payoff-floor opponent constraints, then the threshold constraint is
    pushed/popped for each binary-search probe.  Z3 retains learned conflict
    clauses across probes, making the 2nd+ probes dramatically faster.

    1. Quick SAT filter: can the agent beat their current payoff?
       If UNSAT, skip immediately (costs seconds, not minutes).
    2. Binary search over [witness_payoff, max_payoff] to find the exact
       optimal best-response payoff in ~log2(max_payoff) SAT calls.
    """
    if base_encoding is None:
        raise ValueError("S-IBIS requires base_encoding (cannot use solve_smt_game fallback)")

    from .smt_solve import _extract_solution

    try:
        from z3 import Solver, is_true, sat, unknown
    except Exception as e:  # pragma: no cover
        raise RuntimeError("z3-solver is required for SMT solving") from e

    v = base_encoding.variables

    best_agent: int | None = None
    best_ratio: float = 1.0
    best_delta: int = 0
    best_learned: Policy | None = None
    best_learned_states: int = 0

    for a in range(num_agents):
        # Build ONE solver per agent — reused across all probes
        solver = Solver()
        if timeout_ms is not None:
            solver.set(timeout=timeout_ms)
        solver.add(*base_encoding.constraints)
        _add_payoff_floor_constraints(solver, v, territory, policies, a, base_payoff)

        br_t0 = time.perf_counter()

        # --- Phase 1: Can this agent improve at all? ---
        solver.push()
        solver.add(v.payoff[a] > base_payoff[a])
        check_res = solver.check()

        if check_res != sat:
            solver.pop()
            br_dt = time.perf_counter() - br_t0
            if timing:
                _log(f"iter={it} agent={a} br=no_improvement time_s={br_dt:.3f}")
            else:
                _log(f"iter={it} agent={a} br=no_improvement")
            continue

        # Extract witness from initial check
        model = solver.model()
        check = _extract_solution(model, v, territory, v.A, v.T, True, is_true)
        solver.pop()

        if check.payoff_by_agent is None:
            br_dt = time.perf_counter() - br_t0
            if timing:
                _log(f"iter={it} agent={a} br=no_improvement time_s={br_dt:.3f}")
            else:
                _log(f"iter={it} agent={a} br=no_improvement")
            continue

        # --- Phase 2: Incremental binary search for optimal payoff ---
        best_sol = check
        lo = check.payoff_by_agent[a]  # Known achievable (from SAT witness)
        hi = max_payoff

        search_calls = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2  # Ceiling to avoid infinite loop
            solver.push()
            solver.add(v.payoff[a] > mid - 1)  # payoff > mid-1 ⟺ payoff >= mid
            probe_res = solver.check()
            search_calls += 1

            if probe_res == sat:
                probe_model = solver.model()
                probe = _extract_solution(probe_model, v, territory, v.A, v.T, True, is_true)
                solver.pop()  # Pop BEFORE using the result

                if probe.payoff_by_agent is not None:
                    actual = probe.payoff_by_agent[a]
                    lo = actual  # Skip ahead using witness payoff
                    best_sol = probe
                    if actual >= hi:
                        break
                else:
                    hi = mid - 1
            else:
                solver.pop()
                hi = mid - 1

        # lo = optimal payoff, best_sol has the corresponding model
        new_payoff = lo
        br_dt = time.perf_counter() - br_t0

        if timing:
            _log(f"iter={it} agent={a} br_payoff={new_payoff} search_calls={search_calls} time_s={br_dt:.3f}")
        else:
            _log(f"iter={it} agent={a} br_payoff={new_payoff} search_calls={search_calls}")

        if new_payoff > base_payoff[a]:
            assert best_sol.owner_by_round is not None
            assert best_sol.actions_by_round is not None
            learned = policy_from_solution(best_sol.owner_by_round, best_sol.actions_by_round, a)
            delta = int(new_payoff - base_payoff[a])
            ratio = float(new_payoff) / float(base_payoff[a]) if base_payoff[a] > 0 else float("inf")
            learned_states = len(learned)

            is_better = (
                best_agent is None
                or ratio > best_ratio
                or (ratio == best_ratio and delta > best_delta)
                or (ratio == best_ratio and delta == best_delta and learned_states < best_learned_states)
            )
            if is_better:
                best_agent, best_ratio, best_delta, best_learned, best_learned_states = (
                    a, ratio, delta, learned, learned_states
                )
            _log(f"iter={it} agent={a} improved delta={delta} ratio={ratio:.3f} learned_states={learned_states}")

    return best_agent, best_ratio, best_delta, best_learned, best_learned_states
