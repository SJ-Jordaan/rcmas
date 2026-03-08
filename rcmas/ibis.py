"""Algorithm 1: IBIS (Improvement-Based Iterative Synthesis).

Iterative best-response search for a Nash equilibrium using SMT.
Strategies are state-only deterministic policies with partial explicit
mappings plus an implicit deterministic fallback for unmapped states.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from .model import Coord, Territory
from .smt_solve import BaseEncoding, SmtSolution, build_base_encoding, solve_from_base, solve_smt_game
from .strategy import Policy, StateKey, freeze_profile, policy_from_solution


@dataclass(frozen=True, slots=True)
class IbisResult:
    is_sat: bool
    reason: str
    found_ne: bool
    iterations: int
    strategy: tuple[tuple[Coord | None, ...], ...] | None
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None


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
    """Run IBIS: iterative best-response NE search via SMT (Alg 1)."""

    def _log(msg: str) -> None:
        if progress:
            print(msg, file=sys.stderr, flush=True)

    _ = seed  # reserved for future perturbations

    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if max_iters <= 0:
        raise ValueError("max_iters must be >= 1")

    policies: list[Policy] = [{} for _ in range(num_agents)]
    seen: set[tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]] = set()
    t0 = time.perf_counter()

    def _solver_policies() -> tuple[Policy | None, ...]:
        """Convert empty policies to None for solver compatibility.

        An empty dict forces the deterministic default action (which is
        agent-id-dependent), conflicting with symmetry-breaking constraints.
        None means unconstrained — the solver picks the action freely.
        """
        return tuple(p if p else None for p in policies)

    # Build base encoding once — caches expensive constraint expressions (O(S³))
    base_enc = build_base_encoding(
        territory=territory, num_agents=num_agents, horizon=horizon,
        require_victory=True, symmetry_breaking=symmetry,
        weights=weights, custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target,
        demands=demands,
    )

    for it in range(1, max_iters + 1):
        iter_t0 = time.perf_counter()
        profile_key = freeze_profile(policies)
        if profile_key in seen:
            _log(f"iter={it} cycle_detected")
            base = solve_from_base(
                base_enc,
                objective="sum", fixed_policy_by_agent=_solver_policies(),
                debug=True,
            )
            if timing:
                _log(f"iter={it} final_eval_time_s={time.perf_counter() - iter_t0:.3f}")
            return IbisResult(base.is_sat, "cycle", False, it - 1, base.actions_by_round, base.payoff_by_agent, base)
        seen.add(profile_key)

        base_t0 = time.perf_counter()
        base = solve_from_base(
            base_enc,
            objective="sum", fixed_policy_by_agent=_solver_policies(),
            debug=True, timeout_ms=timeout_ms,
        )
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

        best_agent, best_ratio, best_delta, best_learned, best_learned_states = _find_best_improvement(
            territory=territory, num_agents=num_agents, horizon=horizon,
            policies=policies, base_payoff=base_payoff, timeout_ms=timeout_ms,
            it=it, _log=_log, timing=timing, symmetry=symmetry,
            weights=weights, custom_neighbors=custom_neighbors, weight_balance_target=weight_balance_target,
            demands=demands,
            base_encoding=base_enc,
        )

        if best_agent is None or best_learned is None:
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

    final = solve_from_base(
        base_enc,
        objective="sum", fixed_policy_by_agent=_solver_policies(),
        debug=True, timeout_ms=timeout_ms,
    )
    if timing:
        _log(f"done reason=max_iters iterations={max_iters} total_time_s={time.perf_counter() - t0:.3f}")

    return IbisResult(final.is_sat, "max_iters", False, max_iters, final.actions_by_round, final.payoff_by_agent, final)


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
    """Find the single agent with the best unilateral improvement."""
    best_agent: int | None = None
    best_ratio: float = 1.0
    best_delta: int = 0
    best_learned: Policy | None = None
    best_learned_states: int = 0

    for a in range(num_agents):
        fixed: list[Policy | None] = [None if other == a else (policies[other] if policies[other] else None) for other in range(num_agents)]

        br_t0 = time.perf_counter()
        if base_encoding is not None:
            br = solve_from_base(
                base_encoding,
                objective=a, fixed_policy_by_agent=tuple(fixed),
                enforce_state_only_for_agents=(a,),
                timeout_ms=timeout_ms, debug=True,
            )
        else:
            br = solve_smt_game(
                territory=territory, num_agents=num_agents, horizon=horizon,
                objective=a, fixed_policy_by_agent=tuple(fixed),
                enforce_state_only_for_agents=(a,),
                require_victory=True, timeout_ms=timeout_ms, debug=True,
                symmetry_breaking=symmetry,
                weights=weights, custom_neighbors=custom_neighbors, weight_balance_target=weight_balance_target,
                demands=demands,
            )
        br_dt = time.perf_counter() - br_t0
        if not br.is_sat or br.actions_by_round is None or br.payoff_by_agent is None:
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
