"""Algorithm 2: Q-IBIS (Q-learning-guided IBIS).

Combines SMT-based best-response synthesis with Q-learning proposals
to reduce the search space and improve scalability.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass

from .model import Coord, State, Territory, CollisionError, evolve, largest_region_size, scores
from .qlearning import (
    action_key,
    best_q_action,
    decode_action,
    state_key,
    top_k_actions,
)
from .smt_solve import SmtSolution, solve_smt_game
from .strategy import (
    ActionCandidates,
    Policy,
    StateKey,
    fallback_action,
    freeze_profile,
    policy_action,
    policy_from_solution,
)
from .symmetry import SymmetryInfo, canonical_state, invert_automorphism, symmetry_info


@dataclass(frozen=True, slots=True)
class QibisResult:
    is_sat: bool
    reason: str
    found_ne: bool
    iterations: int
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None
    strategy: tuple[tuple[Coord | None, ...], ...] | None = None


@dataclass(frozen=True, slots=True)
class QibisConfig:
    max_iters: int = 25
    rl_episodes_per_iter: int = 1000
    rl_alpha: float = 0.3
    rl_gamma: float = 0.95
    rl_epsilon_start: float = 0.7
    rl_epsilon_end: float = 0.05
    rl_top_k_actions: int = 3
    rl_persist_q: bool = False
    rl_pretrain_from_smt_base: bool = True
    rl_bootstrap_initial_profile: bool = True
    rl_bootstrap_episodes: int = 800
    defeat_penalty: float = 1000.0
    step_success_reward: float = 1.0
    delta_region_reward: float = 2.0
    terminal_score_reward: float = 1.0
    timeout_ms: int | None = None
    progress: bool = False
    timing: bool = False
    seed: int = 0
    symmetry: bool = False


def _log(progress: bool, msg: str) -> None:
    if progress:
        print(msg, file=sys.stderr, flush=True)


def _epsilon_for(cfg: QibisConfig, episode: int, total_episodes: int) -> float:
    if total_episodes <= 1:
        return cfg.rl_epsilon_end
    t = episode / max(1, total_episodes - 1)
    return cfg.rl_epsilon_start + t * (cfg.rl_epsilon_end - cfg.rl_epsilon_start)


def _evaluate_profile(
    territory: Territory, num_agents: int, horizon: int,
    policies: list[Policy], cfg: QibisConfig,
) -> SmtSolution:
    return solve_smt_game(
        territory=territory, num_agents=num_agents, horizon=horizon,
        objective="sum", fixed_policy_by_agent=tuple(policies),
        require_victory=True, debug=True, timeout_ms=cfg.timeout_ms,
        symmetry_breaking=cfg.symmetry,
    )


def _train_best_response_q(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    agent_id: int,
    opponents: list[Policy],
    cfg: QibisConfig,
    q_init: dict[str, dict[str, float]] | None = None,
    seen_states_init: set[StateKey] | None = None,
    episodes: int | None = None,
    sym: SymmetryInfo | None = None,
) -> tuple[Policy, ActionCandidates, dict[str, dict[str, float]], set[StateKey]]:
    """Train a single agent's best-response Q-table against fixed opponents."""
    q: dict[str, dict[str, float]] = {} if q_init is None else q_init
    seen_states: set[StateKey] = set() if seen_states_init is None else seen_states_init

    rng = random.Random(cfg.seed + 1000 + agent_id)
    total_episodes = cfg.rl_episodes_per_iter if episodes is None else int(episodes)
    if total_episodes <= 0:
        return {}, {}, q, seen_states

    for episode in range(total_episodes):
        eps = _epsilon_for(cfg, episode, total_episodes)
        state = State.initial(territory, num_agents)
        rounds = 0

        while True:
            if rounds >= horizon:
                break
            if state.is_terminal():
                break

            prev_sizes = [largest_region_size(state, i) for i in range(num_agents)]

            actions: dict[int, Coord | None] = {}
            for other in range(num_agents):
                if other == agent_id:
                    continue
                actions[other] = policy_action(opponents[other], state, other)

            options = list(state.available_actions())
            if not options:
                actions[agent_id] = None
                try:
                    next_state = evolve(state, actions)
                    defeated = False
                except CollisionError:
                    defeated = True
                    next_state = state
            else:
                if rng.random() < eps:
                    act = rng.choice(options)
                else:
                    if sym is not None:
                        canon_owner, sigma = canonical_state(state.owner_by_index, sym.automorphisms)
                        inv_sigma = invert_automorphism(sigma)
                        sk = state_key(canon_owner)
                        canon_aks = [str(sigma[int(action_key(state, c))]) for c in options]
                        best = best_q_action(q, sk, canon_aks)
                        if best is None:
                            act = rng.choice(options)
                        else:
                            raw_idx = inv_sigma[int(best)]
                            act = state.ordered_sectors[raw_idx]
                    else:
                        sk = state_key(state.owner_by_index)
                        aks = [action_key(state, c) for c in options]
                        best = best_q_action(q, sk, aks)
                        act = rng.choice(options) if best is None else decode_action(state, best)
                actions[agent_id] = act
                try:
                    next_state = evolve(state, actions)
                    defeated = False
                except CollisionError:
                    defeated = True
                    next_state = state

            # Reward shaping
            if defeated:
                r = -cfg.defeat_penalty
                terminal = True
            else:
                next_sizes = [largest_region_size(next_state, i) for i in range(num_agents)]
                r = 0.0
                if actions.get(agent_id) is not None:
                    r += cfg.step_success_reward
                delta = next_sizes[agent_id] - prev_sizes[agent_id]
                if delta:
                    r += cfg.delta_region_reward * float(delta)
                if next_state.is_terminal():
                    r += cfg.terminal_score_reward * float(scores(next_state)[agent_id])
                terminal = next_state.is_terminal()

            # Q update
            act_chosen = actions.get(agent_id)
            if act_chosen is not None:
                if sym is not None:
                    canon_owner, sigma = canonical_state(state.owner_by_index, sym.automorphisms)
                    s = state_key(canon_owner)
                    a = str(sigma[int(action_key(state, act_chosen))])
                else:
                    s = state_key(state.owner_by_index)
                    a = action_key(state, act_chosen)
                seen_states.add(tuple(state.owner_by_index))

                if terminal:
                    target = r
                else:
                    if sym is not None:
                        next_canon, next_sigma = canonical_state(next_state.owner_by_index, sym.automorphisms)
                        next_sk = state_key(next_canon)
                        next_opts = list(next_state.available_actions())
                        next_aks = [str(next_sigma[int(action_key(next_state, c))]) for c in next_opts]
                    else:
                        next_sk = state_key(next_state.owner_by_index)
                        next_opts = list(next_state.available_actions())
                        next_aks = [action_key(next_state, c) for c in next_opts]
                    best_next = best_q_action(q, next_sk, next_aks)
                    next_max = 0.0 if best_next is None else q.get(next_sk, {}).get(best_next, 0.0)
                    target = r + cfg.rl_gamma * next_max

                old = q.get(s, {}).get(a, 0.0)
                new = (1.0 - cfg.rl_alpha) * old + cfg.rl_alpha * float(target)
                q.setdefault(s, {})[a] = new

            state = next_state
            rounds += 1
            if terminal:
                break

    # Convert to policy + candidates
    policy: Policy = {}
    candidates: ActionCandidates = {}
    for st in seen_states:
        gs = State(territory=territory, num_agents=num_agents, owner_by_index=st, round_index=0)
        options = list(gs.available_actions())
        if not options:
            policy[st] = None
            candidates[st] = (None,)
            continue
        if sym is not None:
            canon_owner, sigma = canonical_state(gs.owner_by_index, sym.automorphisms)
            inv_sigma = invert_automorphism(sigma)
            sk = state_key(canon_owner)
            canon_aks = [str(sigma[int(action_key(gs, c))]) for c in options]
            topk = top_k_actions(q, sk, canon_aks, k=cfg.rl_top_k_actions)
            if not topk:
                policy[st] = options[0]
                candidates[st] = (options[0],)
                continue
            decoded = tuple(gs.ordered_sectors[inv_sigma[int(ak)]] for ak in topk)
        else:
            sk = state_key(gs.owner_by_index)
            aks = [action_key(gs, c) for c in options]
            topk = top_k_actions(q, sk, aks, k=cfg.rl_top_k_actions)
            if not topk:
                policy[st] = options[0]
                candidates[st] = (options[0],)
                continue
            decoded = tuple(decode_action(gs, ak) for ak in topk)
        candidates[st] = decoded
        policy[st] = decoded[0]

    return policy, candidates, q, seen_states


def _pretrain_q_from_smt_trace(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    agent_id: int,
    sol: SmtSolution,
    cfg: QibisConfig,
    q: dict[str, dict[str, float]],
    seen_states: set[StateKey],
    sym: SymmetryInfo | None = None,
) -> None:
    """Replay an SMT solution trace into a Q-table."""
    if sol.owner_by_round is None or sol.actions_by_round is None:
        return
    T = min(horizon, len(sol.actions_by_round), max(0, len(sol.owner_by_round) - 1))
    for t in range(T):
        state = State(territory=territory, num_agents=num_agents, owner_by_index=sol.owner_by_round[t], round_index=t)
        actions_step = sol.actions_by_round[t]
        actions: dict[int, Coord | None] = {i: actions_step[i] for i in range(num_agents)}
        try:
            next_state = evolve(state, actions)
            defeated = False
        except CollisionError:
            defeated = True
            next_state = state

        prev_sizes = [largest_region_size(state, i) for i in range(num_agents)]

        if defeated:
            r = -cfg.defeat_penalty
            terminal = True
        else:
            next_sizes = [largest_region_size(next_state, i) for i in range(num_agents)]
            r = 0.0
            if actions.get(agent_id) is not None:
                r += cfg.step_success_reward
            delta = next_sizes[agent_id] - prev_sizes[agent_id]
            if delta:
                r += cfg.delta_region_reward * float(delta)
            if next_state.is_terminal():
                r += cfg.terminal_score_reward * float(scores(next_state)[agent_id])
            terminal = next_state.is_terminal()

        act_chosen = actions.get(agent_id)
        if act_chosen is None:
            continue

        if sym is not None:
            canon_owner, sigma = canonical_state(state.owner_by_index, sym.automorphisms)
            s = state_key(canon_owner)
            a = str(sigma[int(action_key(state, act_chosen))])
        else:
            s = state_key(state.owner_by_index)
            a = action_key(state, act_chosen)
        seen_states.add(tuple(state.owner_by_index))

        if terminal:
            target = r
        else:
            if sym is not None:
                next_canon, next_sigma = canonical_state(next_state.owner_by_index, sym.automorphisms)
                next_sk = state_key(next_canon)
                next_opts = list(next_state.available_actions())
                next_aks = [str(next_sigma[int(action_key(next_state, c))]) for c in next_opts]
            else:
                next_sk = state_key(next_state.owner_by_index)
                next_opts = list(next_state.available_actions())
                next_aks = [action_key(next_state, c) for c in next_opts]
            best_next = best_q_action(q, next_sk, next_aks)
            next_max = 0.0 if best_next is None else q.get(next_sk, {}).get(best_next, 0.0)
            target = r + cfg.rl_gamma * next_max

        old = q.get(s, {}).get(a, 0.0)
        new = (1.0 - cfg.rl_alpha) * old + cfg.rl_alpha * float(target)
        q.setdefault(s, {})[a] = new


def _bootstrap_profile_from_q(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    q_by_agent: list[dict[str, dict[str, float]]],
    sym: SymmetryInfo | None = None,
) -> list[Policy]:
    """Derive a collision-free initial policy profile from Q-tables via greedy rollout."""
    policies: list[Policy] = [{} for _ in range(num_agents)]
    state = State.initial(territory, num_agents)
    rounds = 0

    while rounds < horizon and not state.is_terminal():
        options = list(state.available_actions())
        if not options:
            break

        chosen_by_agent: dict[int, Coord | None] = {}
        taken: set[Coord] = set()
        prev_action_idx = -1

        for a in range(num_agents):
            if sym is not None:
                canon_owner, sigma = canonical_state(state.owner_by_index, sym.automorphisms)
                inv_sigma = invert_automorphism(sigma)
                sk = state_key(canon_owner)
                canon_aks = [str(sigma[int(action_key(state, c))]) for c in options]
                ranked = top_k_actions(q_by_agent[a], sk, canon_aks, k=len(canon_aks))
                if ranked:
                    ranked_coords = [state.ordered_sectors[inv_sigma[int(ak)]] for ak in ranked]
                else:
                    ranked_coords = list(options)
            else:
                sk = state_key(state.owner_by_index)
                aks = [action_key(state, c) for c in options]
                ranked = top_k_actions(q_by_agent[a], sk, aks, k=len(aks))
                ranked_coords = [decode_action(state, ak) for ak in ranked] if ranked else list(options)

            # Symmetry-constrained filtering for round 0
            if sym is not None and rounds == 0:
                if a == 0:
                    valid_indices = set(sym.representatives)
                    ranked_coords = [c for c in ranked_coords
                                     if territory.index_of(c) in valid_indices]
                else:
                    ranked_coords = [c for c in ranked_coords
                                     if territory.index_of(c) is not None
                                     and territory.index_of(c) >= prev_action_idx]

            pick = None
            for c in ranked_coords:
                if c not in taken:
                    pick = c
                    break

            chosen_by_agent[a] = pick
            if pick is not None:
                taken.add(pick)
                if sym is not None and rounds == 0:
                    prev_action_idx = territory.index_of(pick)

        key: StateKey = tuple(state.owner_by_index)
        for a in range(num_agents):
            policies[a][key] = chosen_by_agent[a]

        try:
            next_state = evolve(state, chosen_by_agent)
        except CollisionError:
            break
        state = next_state
        rounds += 1

    return policies


def solve_qibis(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    cfg: QibisConfig,
) -> QibisResult:
    """Run Q-IBIS: Q-learning-guided iterative best-response (Alg 2)."""
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if cfg.max_iters <= 0:
        raise ValueError("max_iters must be >= 1")

    sym = symmetry_info(territory) if cfg.symmetry else None

    policies: list[Policy] = [{} for _ in range(num_agents)]
    seen_profiles: set[tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]] = set()

    q_by_agent: list[dict[str, dict[str, float]] | None] = [None] * num_agents
    seen_states_by_agent: list[set[StateKey] | None] = [None] * num_agents

    # RL bootstrap
    if cfg.rl_episodes_per_iter > 0 and cfg.rl_bootstrap_initial_profile and cfg.rl_bootstrap_episodes > 0:
        _log(cfg.progress, f"bootstrap rl_episodes={cfg.rl_bootstrap_episodes}")
        q_boot: list[dict[str, dict[str, float]]] = []
        for a in range(num_agents):
            _, _, q_next, seen_next = _train_best_response_q(
                territory=territory, num_agents=num_agents, horizon=horizon,
                agent_id=a, opponents=policies, cfg=cfg,
                episodes=cfg.rl_bootstrap_episodes, sym=sym,
            )
            q_boot.append(q_next)
            if cfg.rl_persist_q:
                q_by_agent[a] = q_next
                seen_states_by_agent[a] = seen_next

        policies = _bootstrap_profile_from_q(
            territory=territory, num_agents=num_agents, horizon=horizon,
            q_by_agent=q_boot, sym=sym,
        )
        _log(cfg.progress, f"bootstrap policy_sizes={[len(p) for p in policies]}")

    t0 = time.perf_counter()

    for it in range(1, cfg.max_iters + 1):
        iter_t0 = time.perf_counter()
        key = freeze_profile(policies)
        if key in seen_profiles:
            base = _evaluate_profile(territory, num_agents, horizon, policies, cfg)
            if cfg.timing:
                _log(cfg.progress, f"done reason=cycle iterations={it - 1} total_time_s={time.perf_counter() - t0:.3f}")
            return QibisResult(base.is_sat, "cycle", False, it - 1, base.payoff_by_agent, base, base.actions_by_round)
        seen_profiles.add(key)

        base_eval_t0 = time.perf_counter()
        base = _evaluate_profile(territory, num_agents, horizon, policies, cfg)
        base_eval_dt = time.perf_counter() - base_eval_t0

        if not base.is_sat:
            if cfg.timing:
                _log(cfg.progress, f"iter={it} base_eval reason={base.reason} time_s={base_eval_dt:.3f}")
            return QibisResult(False, base.reason, False, it - 1, None, base, base.actions_by_round)
        if base.payoff_by_agent is None:
            return QibisResult(False, "missing_debug_payoff", False, it - 1, None, base, base.actions_by_round)

        base_payoff = base.payoff_by_agent
        if cfg.timing:
            _log(cfg.progress, f"iter={it} base_payoff={base_payoff} base_eval_time_s={base_eval_dt:.3f}")
        else:
            _log(cfg.progress, f"iter={it} base_payoff={base_payoff}")

        best_agent: int | None = None
        best_ratio: float = 1.0
        best_delta: int = 0
        best_learned: Policy | None = None
        best_learned_states: int = 0

        candidates_by_agent: list[ActionCandidates | None] = [None] * num_agents

        for a in range(num_agents):
            # 1) RL proposal
            if cfg.rl_episodes_per_iter > 0:
                rl_t0 = time.perf_counter()
                q_seed = q_by_agent[a] if cfg.rl_persist_q else {}
                seen_seed = seen_states_by_agent[a] if cfg.rl_persist_q else set()

                if cfg.rl_pretrain_from_smt_base:
                    _pretrain_q_from_smt_trace(
                        territory=territory, num_agents=num_agents, horizon=horizon,
                        agent_id=a, sol=base, cfg=cfg,
                        q=q_seed if q_seed is not None else {},
                        seen_states=seen_seed if seen_seed is not None else set(),
                        sym=sym,
                    )

                proposal, candidates, q_next, seen_states_next = _train_best_response_q(
                    territory=territory, num_agents=num_agents, horizon=horizon,
                    agent_id=a, opponents=policies, cfg=cfg,
                    q_init=q_seed, seen_states_init=seen_seed, sym=sym,
                )
                candidates_by_agent[a] = candidates
                if cfg.rl_persist_q:
                    q_by_agent[a] = q_next
                    seen_states_by_agent[a] = seen_states_next
                rl_dt = time.perf_counter() - rl_t0
                if cfg.timing:
                    _log(cfg.progress, f"iter={it} agent={a} rl_states={len(proposal)} rl_time_s={rl_dt:.3f}")
                else:
                    _log(cfg.progress, f"iter={it} agent={a} rl_states={len(proposal)}")

            # 2) SMT best-response
            fixed: list[Policy | None] = [None if other == a else policies[other] for other in range(num_agents)]
            action_cands = None
            cand = candidates_by_agent[a]
            if cand is not None and cfg.rl_top_k_actions > 0:
                action_cands = tuple(cand if i == a else None for i in range(num_agents))

            br_t0 = time.perf_counter()
            br = solve_smt_game(
                territory=territory, num_agents=num_agents, horizon=horizon,
                objective=a, fixed_policy_by_agent=tuple(fixed),
                action_candidates_by_agent=action_cands,
                enforce_state_only_for_agents=(a,),
                require_victory=True, debug=True, timeout_ms=cfg.timeout_ms,
                symmetry_breaking=cfg.symmetry,
            )
            br_dt = time.perf_counter() - br_t0

            if not br.is_sat:
                if cfg.timing:
                    _log(cfg.progress, f"iter={it} agent={a} smt_br reason={br.reason} time_s={br_dt:.3f}")
                else:
                    _log(cfg.progress, f"iter={it} agent={a} smt_br reason={br.reason}")
                if br.reason == "unknown":
                    return QibisResult(False, "unknown", False, it - 1, base_payoff, base, base.actions_by_round)
                continue

            if br.payoff_by_agent is None:
                return QibisResult(False, "missing_debug_payoff", False, it - 1, base_payoff, br, br.actions_by_round)

            new_payoff = br.payoff_by_agent[a]
            if cfg.timing:
                _log(cfg.progress, f"iter={it} agent={a} smt_br_payoff={new_payoff} time_s={br_dt:.3f}")
            else:
                _log(cfg.progress, f"iter={it} agent={a} smt_br_payoff={new_payoff}")

            if new_payoff > base_payoff[a]:
                assert br.owner_by_round is not None and br.actions_by_round is not None
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
                _log(cfg.progress, f"iter={it} agent={a} improved delta={delta} ratio={ratio:.3f} learned_states={learned_states}")

        if best_agent is None or best_learned is None:
            if cfg.timing:
                _log(cfg.progress, f"iter={it} converged_time_s={time.perf_counter() - iter_t0:.3f}")
                _log(cfg.progress, f"done reason=ne iterations={it} total_time_s={time.perf_counter() - t0:.3f}")
            return QibisResult(True, "ne", True, it, base_payoff, base, base.actions_by_round)

        updated = dict(policies[best_agent])
        updated.update(best_learned)
        policies = [dict(p) for p in policies]
        policies[best_agent] = updated

        _log(cfg.progress, f"iter={it} picked_agent={best_agent} picked_delta={best_delta} picked_ratio={best_ratio:.3f} picked_learned_states={best_learned_states}")
        if cfg.timing:
            _log(cfg.progress, f"iter={it} policy_sizes={[len(p) for p in policies]} iter_time_s={time.perf_counter() - iter_t0:.3f}")
        else:
            _log(cfg.progress, f"iter={it} policy_sizes={[len(p) for p in policies]}")

    final = _evaluate_profile(territory, num_agents, horizon, policies, cfg)
    if cfg.timing:
        _log(cfg.progress, f"done reason=max_iters iterations={cfg.max_iters} total_time_s={time.perf_counter() - t0:.3f}")

    return QibisResult(final.is_sat, "max_iters", False, cfg.max_iters, final.payoff_by_agent, final, final.actions_by_round)
