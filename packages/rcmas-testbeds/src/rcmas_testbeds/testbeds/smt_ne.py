from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Iterable

from rcmas_core.engine import Coord, Territory

from .base import Testbed
from .smt_co import SmtSolution, solve_smt_game


@dataclass(frozen=True, slots=True)
class SmtNaiveNEResult:
    is_sat: bool
    reason: str
    found_ne: bool
    iterations: int
    # For convenience/debugging: realized joint trace under the final policy profile.
    # This is not the policy itself.
    strategy: tuple[tuple[Coord | None, ...], ...] | None
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None


StateKey = tuple[int, ...]
Policy = dict[StateKey, Coord | None]


def _freeze_policy(policy: Policy) -> tuple[tuple[StateKey, tuple[int, int] | None], ...]:
    items: list[tuple[StateKey, tuple[int, int] | None]] = []
    for state, action in policy.items():
        items.append((state, None if action is None else (action.x, action.y)))
    items.sort()
    return tuple(items)


def _freeze_profile(policies: Iterable[Policy]) -> tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]:
    return tuple(_freeze_policy(p) for p in policies)


def _policy_from_solution_for_agent(sol: SmtSolution, agent_id: int) -> Policy:
    if sol.owner_by_round is None or sol.actions_by_round is None:
        raise ValueError("solution missing debug owner/actions")

    out: Policy = {}
    for t in range(len(sol.actions_by_round)):
        state = sol.owner_by_round[t]
        action = sol.actions_by_round[t][agent_id]
        out[state] = action
    return out


def solve_naive_ne(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    seed: int = 0,
    max_iters: int = 25,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
) -> SmtNaiveNEResult:
    """Iterative best-response search for a (naive) Nash equilibrium.

    Strategies are *state-only deterministic policies* with partial explicit mappings
    plus an implicit deterministic fallback for any unmapped state.

    Correctness intent:
    - When optimizing agent i, all other agents are constrained to follow their fixed
      policies (explicit mappings + fallback) so the solver cannot "complete" their
      missing strategy in favor of i.
    - The optimizing agent is free to deviate, but we also enforce that its chosen
      actions are consistent with a state-only policy (same state => same action)
      across the horizon.
    """

    def _log(msg: str) -> None:
        if progress:
            print(msg, file=sys.stderr, flush=True)

    _ = seed  # reserved for future perturbations/tie-breaking

    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if max_iters <= 0:
        raise ValueError("max_iters must be >= 1")

    sectors = territory.ordered_sectors()
    if num_agents * horizon < len(sectors):
        return SmtNaiveNEResult(False, "horizon too small to guarantee victory", False, 0, None, None, None)

    # Start with empty explicit policies for all agents; the solver-enforced fallback
    # defines the behavior for all unmapped states.
    policies: list[Policy] = [{} for _ in range(num_agents)]

    seen: set[tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]] = set()

    t0 = time.perf_counter()

    for it in range(1, max_iters + 1):
        iter_t0 = time.perf_counter()
        profile_key = _freeze_profile(policies)
        if profile_key in seen:
            _log(f"iter={it} cycle_detected")
            base = solve_smt_game(
                territory=territory,
                num_agents=num_agents,
                horizon=horizon,
                objective="sum",
                fixed_actions_by_round=None,
                fixed_policy_by_agent=tuple(policies),
                require_victory=True,
                debug=True,
            )
            if timing:
                _log(f"iter={it} final_eval_time_s={time.perf_counter() - iter_t0:.3f}")
            return SmtNaiveNEResult(base.is_sat, "cycle", False, it - 1, base.actions_by_round, base.payoff_by_agent, base)
        seen.add(profile_key)

        base_t0 = time.perf_counter()
        base = solve_smt_game(
            territory=territory,
            num_agents=num_agents,
            horizon=horizon,
            objective="sum",
            fixed_actions_by_round=None,
            fixed_policy_by_agent=tuple(policies),
            require_victory=True,
            debug=True,
            timeout_ms=timeout_ms,
        )
        base_dt = time.perf_counter() - base_t0
        if not base.is_sat:
            return SmtNaiveNEResult(False, base.reason, False, it - 1, base.actions_by_round, None, base)
        if base.payoff_by_agent is None:
            return SmtNaiveNEResult(False, "missing_debug_payoff", False, it - 1, base.actions_by_round, None, base)

        base_payoff = base.payoff_by_agent

        if timing:
            _log(f"iter={it} base_payoff={base_payoff} base_solve_time_s={base_dt:.3f}")
        else:
            _log(f"iter={it} base_payoff={base_payoff}")

        improved = False
        next_policies: list[Policy] = [dict(p) for p in policies]

        for a in range(num_agents):
            fixed_policy_by_agent: list[Policy | None] = []
            for other in range(num_agents):
                fixed_policy_by_agent.append(None if other == a else policies[other])

            br_t0 = time.perf_counter()
            br = solve_smt_game(
                territory=territory,
                num_agents=num_agents,
                horizon=horizon,
                objective=a,
                fixed_actions_by_round=None,
                fixed_policy_by_agent=tuple(fixed_policy_by_agent),
                enforce_state_only_for_agents=(a,),
                require_victory=True,
                timeout_ms=timeout_ms,
                debug=True,
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
                improved = True
                learned = _policy_from_solution_for_agent(br, a)
                updated = dict(next_policies[a])
                updated.update(learned)
                next_policies[a] = updated
                _log(f"iter={it} agent={a} improved delta={new_payoff - base_payoff[a]} learned_states={len(learned)} total_states={len(updated)}")

        if not improved:
            if timing:
                _log(f"iter={it} converged_time_s={time.perf_counter() - iter_t0:.3f}")
                _log(f"done reason=ne iterations={it} total_time_s={time.perf_counter() - t0:.3f}")
            return SmtNaiveNEResult(True, "ne", True, it, base.actions_by_round, base_payoff, base)

        policies = next_policies

        if timing:
            _log(f"iter={it} updated_profile_states={[len(p) for p in policies]} iter_time_s={time.perf_counter() - iter_t0:.3f}")
        else:
            _log(f"iter={it} updated_profile_states={[len(p) for p in policies]}")

    final = solve_smt_game(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        objective="sum",
        fixed_actions_by_round=None,
        fixed_policy_by_agent=tuple(policies),
        require_victory=True,
        debug=True,
        timeout_ms=timeout_ms,
    )

    if timing:
        _log(f"done reason=max_iters iterations={max_iters} total_time_s={time.perf_counter() - t0:.3f}")

    return SmtNaiveNEResult(
        final.is_sat,
        "max_iters",
        False,
        max_iters,
        final.actions_by_round,
        final.payoff_by_agent,
        final,
    )


@dataclass(frozen=True, slots=True)
class SmtNaiveNETestbed(Testbed):
    name: str = "smt-ne"
    seed: int = 0
    max_iters: int = 25
    progress: bool = False
    timing: bool = False
    timeout_ms: int | None = None

    def build_agents(self, *, territory: Territory, num_agents: int, max_rounds: int) -> list[object]:  # noqa: ARG002
        raise NotImplementedError("smt-ne does not run via GameEngine; use solve_naive_ne")

    def solve(self, *, territory: Territory, num_agents: int, horizon: int) -> SmtNaiveNEResult:
        return solve_naive_ne(
            territory=territory,
            num_agents=num_agents,
            horizon=horizon,
            seed=self.seed,
            max_iters=self.max_iters,
            progress=self.progress,
            timing=self.timing,
            timeout_ms=self.timeout_ms,
        )
