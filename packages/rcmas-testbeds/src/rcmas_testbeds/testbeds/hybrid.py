from __future__ import annotations

import base64
import random
import sys
import time
from dataclasses import dataclass

from rcmas_core.engine import Coord, GameState, GameStatus, Territory

from .base import Testbed
from .smt_co import SmtSolution, solve_smt_game


StateKey = tuple[int, ...]
Policy = dict[StateKey, Coord | None]
ActionCandidates = dict[StateKey, tuple[Coord | None, ...]]


@dataclass(frozen=True, slots=True)
class HybridNEResult:
    is_sat: bool
    reason: str
    found_ne: bool
    iterations: int
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None
    # Convenience/debug: realized joint trace under the final policy profile.
    strategy: tuple[tuple[Coord | None, ...], ...] | None = None


@dataclass(frozen=True, slots=True)
class HybridConfig:
    # Outer loop: best-response iterations
    max_iters: int = 25

    # RL: episodes per agent per outer iteration (0 disables RL proposals)
    rl_episodes_per_iter: int = 200
    rl_alpha: float = 0.3
    rl_gamma: float = 0.95
    rl_epsilon_start: float = 0.7
    rl_epsilon_end: float = 0.05

    # When proposing actions via RL, restrict SMT best-response to the RL top-k
    # actions for visited states (plus -1 as a safety escape hatch).
    rl_top_k_actions: int = 3

    # RL warm-start: keep Q-tables across outer iterations.
    rl_persist_q: bool = False

    # Bootstrap: before the first SMT iteration, do extra RL training (per-agent)
    # against the default fallback opponents, then derive an initial collision-free
    # partial policy profile by simulating a joint greedy rollout.
    rl_bootstrap_initial_profile: bool = True
    rl_bootstrap_episodes: int = 800

    # Reward shaping (matches qlearning defaults, but single-agent view)
    defeat_penalty: float = 1000.0
    step_success_reward: float = 1.0
    delta_region_reward: float = 2.0
    terminal_score_reward: float = 1.0

    # SMT
    timeout_ms: int | None = None

    # Output
    progress: bool = False
    timing: bool = False

    # RNG seed
    seed: int = 0


def _log(progress: bool, msg: str) -> None:
    if progress:
        print(msg, file=sys.stderr, flush=True)


def _epsilon_for_episode(cfg: HybridConfig, episode: int) -> float:
    if cfg.rl_episodes_per_iter <= 1:
        return cfg.rl_epsilon_end
    t = episode / max(1, cfg.rl_episodes_per_iter - 1)
    return cfg.rl_epsilon_start + t * (cfg.rl_epsilon_end - cfg.rl_epsilon_start)


def _epsilon_for_episode_counts(cfg: HybridConfig, episode: int, total_episodes: int) -> float:
    if total_episodes <= 1:
        return cfg.rl_epsilon_end
    t = episode / max(1, total_episodes - 1)
    return cfg.rl_epsilon_start + t * (cfg.rl_epsilon_end - cfg.rl_epsilon_start)


def _state_key(owner_by_index: tuple[int, ...]) -> str:
    # Map owners: -1 -> 0, agent0->1, ...
    raw = bytes((o + 1) if o >= 0 else 0 for o in owner_by_index)
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _action_key(state: GameState, action: Coord) -> str:
    idx = state.territory.index_of(action)
    if idx is None:
        raise ValueError("action not in territory")
    return str(idx)


def _decode_action(state: GameState, action_key: str) -> Coord:
    idx = int(action_key)
    return state.ordered_sectors[idx]


def _best_action(q: dict[str, dict[str, float]], sk: str, action_keys: list[str]) -> str | None:
    if not action_keys:
        return None
    best = action_keys[0]
    best_v = q.get(sk, {}).get(best, 0.0)
    for ak in action_keys[1:]:
        v = q.get(sk, {}).get(ak, 0.0)
        if v > best_v:
            best, best_v = ak, v
    return best


def _top_k_actions(
    q: dict[str, dict[str, float]],
    sk: str,
    action_keys: list[str],
    *,
    k: int,
) -> list[str]:
    if not action_keys or k <= 0:
        return []
    scored = [(q.get(sk, {}).get(ak, 0.0), ak) for ak in action_keys]
    # Highest Q first, stable tie-break by action key.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [ak for _, ak in scored[: min(k, len(scored))]]


def _fallback_action(state: GameState, agent_id: int) -> Coord | None:
    # Match the SMT fallback: choose the agent_id-th unowned sector in ordered order.
    options = state.available_sectors()
    if agent_id < 0 or agent_id >= state.num_agents:
        raise ValueError("agent_id out of range")
    if agent_id >= len(options):
        return None
    return options[agent_id]


def _policy_action(policy: Policy, state: GameState, agent_id: int) -> Coord | None:
    key: StateKey = tuple(state.owner_by_index)
    chosen = policy.get(key)
    if chosen is None:
        return _fallback_action(state, agent_id)

    # Safety: if a stored choice is no longer legal, degrade to fallback.
    if chosen not in state.available_sectors():
        return _fallback_action(state, agent_id)
    return chosen


def _evaluate_profile(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    policies: list[Policy],
    cfg: HybridConfig,
) -> SmtSolution:
    return solve_smt_game(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        objective="sum",
        fixed_actions_by_round=None,
        fixed_policy_by_agent=tuple(policies),
        require_victory=True,
        debug=True,
        timeout_ms=cfg.timeout_ms,
    )


def _policy_from_solution_for_agent(sol: SmtSolution, agent_id: int) -> Policy:
    if sol.owner_by_round is None or sol.actions_by_round is None:
        raise ValueError("solution missing debug owner/actions")

    out: Policy = {}
    for t in range(len(sol.actions_by_round)):
        state_key: StateKey = sol.owner_by_round[t]
        out[state_key] = sol.actions_by_round[t][agent_id]
    return out


def _train_best_response_q(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    agent_id: int,
    opponents: list[Policy],
    cfg: HybridConfig,
    q_init: dict[str, dict[str, float]] | None = None,
    seen_states_init: set[StateKey] | None = None,
    episodes: int | None = None,
) -> tuple[Policy, ActionCandidates, dict[str, dict[str, float]], set[StateKey]]:
    # Q-table: q[state_key][action_key] = value
    q: dict[str, dict[str, float]] = {} if q_init is None else q_init
    seen_states: set[StateKey] = set() if seen_states_init is None else seen_states_init

    rng = random.Random(cfg.seed + 1000 + agent_id)

    total_episodes = cfg.rl_episodes_per_iter if episodes is None else int(episodes)
    if total_episodes <= 0:
        return {}, {}, q, seen_states

    for episode in range(total_episodes):
        eps = _epsilon_for_episode_counts(cfg, episode, total_episodes)
        state = GameState.new(territory, num_agents)
        rounds = 0

        while True:
            if rounds >= horizon:
                # Treat hitting horizon as defeat (no full acquisition).
                reward = -cfg.defeat_penalty
                break
            if state.is_terminal():
                reward = cfg.terminal_score_reward * float(state.scores()[agent_id])
                break

            prev_sizes = [state.largest_region_size(i) for i in range(num_agents)]

            actions: dict[int, Coord | None] = {}
            # opponents act deterministically from their policies
            for other in range(num_agents):
                if other == agent_id:
                    continue
                actions[other] = _policy_action(opponents[other], state, other)

            # learning agent action
            options = list(state.available_sectors())
            if not options:
                actions[agent_id] = None
                next_state, outcome = state.step(actions)
            else:
                if rng.random() < eps:
                    act = rng.choice(options)
                else:
                    sk = _state_key(state.owner_by_index)
                    aks = [_action_key(state, c) for c in options]
                    best = _best_action(q, sk, aks)
                    act = rng.choice(options) if best is None else _decode_action(state, best)
                actions[agent_id] = act
                next_state, outcome = state.step(actions)

            # reward shaping for the learning agent only
            if outcome.status == GameStatus.DEFEAT:
                r = -cfg.defeat_penalty
                terminal = True
            else:
                next_sizes = [next_state.largest_region_size(i) for i in range(num_agents)]
                r = 0.0
                if actions.get(agent_id) is not None:
                    r += cfg.step_success_reward
                delta = next_sizes[agent_id] - prev_sizes[agent_id]
                if delta:
                    r += cfg.delta_region_reward * float(delta)
                if next_state.is_terminal():
                    r += cfg.terminal_score_reward * float(next_state.scores()[agent_id])
                terminal = next_state.is_terminal()

            # Q update
            act_chosen = actions.get(agent_id)
            if act_chosen is not None:
                s = _state_key(state.owner_by_index)
                a = _action_key(state, act_chosen)
                seen_states.add(tuple(state.owner_by_index))

                if terminal:
                    target = r
                else:
                    next_sk = _state_key(next_state.owner_by_index)
                    next_opts = list(next_state.available_sectors())
                    next_aks = [_action_key(next_state, c) for c in next_opts]
                    best_next = _best_action(q, next_sk, next_aks)
                    next_max = 0.0 if best_next is None else q.get(next_sk, {}).get(best_next, 0.0)
                    target = r + cfg.rl_gamma * next_max

                old = q.get(s, {}).get(a, 0.0)
                new = (1.0 - cfg.rl_alpha) * old + cfg.rl_alpha * float(target)
                q.setdefault(s, {})[a] = new

            state = next_state
            rounds += 1

            if terminal:
                break

    # Convert to a partial deterministic policy over visited states.
    policy: Policy = {}
    candidates: ActionCandidates = {}
    for st in seen_states:
        gs = GameState(territory=territory, num_agents=num_agents, owner_by_index=st, round_index=0)
        options = list(gs.available_sectors())
        if not options:
            policy[st] = None
            candidates[st] = (None,)
            continue

        sk = _state_key(gs.owner_by_index)
        aks = [_action_key(gs, c) for c in options]
        topk = _top_k_actions(q, sk, aks, k=cfg.rl_top_k_actions)
        if not topk:
            # No learned preference: keep deterministic choice for reproducibility.
            policy[st] = options[0]
            candidates[st] = (options[0],)
            continue

        decoded = tuple(_decode_action(gs, ak) for ak in topk)
        candidates[st] = decoded
        policy[st] = decoded[0]

    return policy, candidates, q, seen_states


def _bootstrap_profile_from_q(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    q_by_agent: list[dict[str, dict[str, float]]],
) -> list[Policy]:
    """Derive a collision-free partial policy profile by joint greedy rollout.

    Each step, agents are assigned distinct sectors using their Q-induced ranking.
    This produces per-agent state->action entries that are safe under simultaneous
    execution for the visited states.
    """

    policies: list[Policy] = [{} for _ in range(num_agents)]
    state = GameState.new(territory, num_agents)
    rounds = 0

    while rounds < horizon and not state.is_terminal():
        options = list(state.available_sectors())
        if not options:
            break

        chosen_by_agent: dict[int, Coord | None] = {}
        taken: set[Coord] = set()

        for a in range(num_agents):
            sk = _state_key(state.owner_by_index)
            aks = [_action_key(state, c) for c in options]
            ranked = _top_k_actions(q_by_agent[a], sk, aks, k=len(aks))

            # Convert ranking to coords; if unseen, fall back to deterministic ordering.
            ranked_coords = [_decode_action(state, ak) for ak in ranked] if ranked else list(options)

            pick = None
            for c in ranked_coords:
                if c not in taken:
                    pick = c
                    break
            if pick is None:
                pick = None

            chosen_by_agent[a] = pick
            if pick is not None:
                taken.add(pick)

        # Record this state's joint decision as a partial policy profile.
        key: StateKey = tuple(state.owner_by_index)
        for a in range(num_agents):
            policies[a][key] = chosen_by_agent[a]

        next_state, outcome = state.step(chosen_by_agent)
        if outcome.status == GameStatus.DEFEAT:
            # Shouldn't happen because we enforce distinct picks, but stay safe.
            break
        state = next_state
        rounds += 1

    return policies


def solve_hybrid_ne(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    cfg: HybridConfig,
) -> HybridNEResult:
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if cfg.max_iters <= 0:
        raise ValueError("max_iters must be >= 1")

    policies: list[Policy] = [{} for _ in range(num_agents)]
    seen_profiles: set[tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]] = set()

    # Optional persistent Q-learning state (warm-start across iterations).
    q_by_agent: list[dict[str, dict[str, float]] | None] = [None for _ in range(num_agents)]
    seen_states_by_agent: list[set[StateKey] | None] = [None for _ in range(num_agents)]

    # RL bootstrap: train per-agent vs fallback opponents and derive a collision-free
    # initial partial policy profile (similar intent to "good initial strategies").
    if cfg.rl_episodes_per_iter > 0 and cfg.rl_bootstrap_initial_profile and cfg.rl_bootstrap_episodes > 0:
        _log(cfg.progress, f"bootstrap rl_episodes={cfg.rl_bootstrap_episodes}")
        q_boot: list[dict[str, dict[str, float]]] = []
        for a in range(num_agents):
            _, _, q_next, seen_next = _train_best_response_q(
                territory=territory,
                num_agents=num_agents,
                horizon=horizon,
                agent_id=a,
                opponents=policies,
                cfg=cfg,
                q_init=None,
                seen_states_init=None,
                episodes=cfg.rl_bootstrap_episodes,
            )
            q_boot.append(q_next)
            if cfg.rl_persist_q:
                q_by_agent[a] = q_next
                seen_states_by_agent[a] = seen_next

        policies = _bootstrap_profile_from_q(territory=territory, num_agents=num_agents, horizon=horizon, q_by_agent=q_boot)
        _log(cfg.progress, f"bootstrap policy_sizes={[len(p) for p in policies]}")

    def freeze_policy(p: Policy) -> tuple[tuple[StateKey, tuple[int, int] | None], ...]:
        items: list[tuple[StateKey, tuple[int, int] | None]] = []
        for state, action in p.items():
            items.append((state, None if action is None else (action.x, action.y)))
        items.sort()
        return tuple(items)

    def freeze_profile(ps: list[Policy]) -> tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]:
        return tuple(freeze_policy(p) for p in ps)

    t0 = time.perf_counter()

    for it in range(1, cfg.max_iters + 1):
        iter_t0 = time.perf_counter()
        key = freeze_profile(policies)
        if key in seen_profiles:
            base = _evaluate_profile(territory=territory, num_agents=num_agents, horizon=horizon, policies=policies, cfg=cfg)
            if cfg.timing:
                _log(cfg.progress, f"done reason=cycle iterations={it - 1} total_time_s={time.perf_counter() - t0:.3f}")
            return HybridNEResult(base.is_sat, "cycle", False, it - 1, base.payoff_by_agent, base, base.actions_by_round)
        seen_profiles.add(key)

        base_eval_t0 = time.perf_counter()
        base = _evaluate_profile(territory=territory, num_agents=num_agents, horizon=horizon, policies=policies, cfg=cfg)
        base_eval_dt = time.perf_counter() - base_eval_t0

        if not base.is_sat:
            # Could be unsat or unknown (timeout). Preserve reason.
            if cfg.timing:
                _log(cfg.progress, f"iter={it} base_eval reason={base.reason} time_s={base_eval_dt:.3f}")
            return HybridNEResult(False, base.reason, False, it - 1, None, base, base.actions_by_round)

        if base.payoff_by_agent is None:
            return HybridNEResult(False, "missing_debug_payoff", False, it - 1, None, base, base.actions_by_round)

        base_payoff = base.payoff_by_agent
        if cfg.timing:
            _log(cfg.progress, f"iter={it} base_payoff={base_payoff} base_eval_time_s={base_eval_dt:.3f}")
        else:
            _log(cfg.progress, f"iter={it} base_payoff={base_payoff}")

        # Important: do not merge multiple unilateral best-responses in one
        # iteration. Combining independently learned policies can introduce
        # collisions and make the next baseline evaluation UNSAT. Instead, pick
        # the single agent with the best improvement and update only that agent.
        best_agent: int | None = None
        best_ratio: float = 1.0
        best_delta: int = 0
        best_learned: Policy | None = None
        best_learned_states: int = 0

        # RL candidates for this iteration, used to restrict SMT BR solves.
        # Only populated when RL is enabled.
        candidates_by_agent: list[ActionCandidates | None] = [None for _ in range(num_agents)]

        for a in range(num_agents):
            # 1) RL proposal (optional)
            if cfg.rl_episodes_per_iter > 0:
                rl_t0 = time.perf_counter()
                proposal, candidates, q_next, seen_states_next = _train_best_response_q(
                    territory=territory,
                    num_agents=num_agents,
                    horizon=horizon,
                    agent_id=a,
                    opponents=policies,
                    cfg=cfg,
                    q_init=q_by_agent[a] if cfg.rl_persist_q else None,
                    seen_states_init=seen_states_by_agent[a] if cfg.rl_persist_q else None,
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
                # Important: do NOT merge RL's proposed policy into the fixed-policy
                # profile. RL proposals are not collision-aware w.r.t. simultaneous
                # moves, so fixing them can easily make the baseline evaluation UNSAT.
                # We only use RL output to restrict the deviator's SMT action space.

            # 2) SMT verification/repair: exact best response against fixed opponents
            fixed_policy_by_agent: list[Policy | None] = []
            for other in range(num_agents):
                fixed_policy_by_agent.append(None if other == a else policies[other])

            action_candidates_by_agent = None
            cand = candidates_by_agent[a]
            if cand is not None and cfg.rl_top_k_actions > 0:
                action_candidates_by_agent = tuple(cand if i == a else None for i in range(num_agents))

            br_t0 = time.perf_counter()
            br = solve_smt_game(
                territory=territory,
                num_agents=num_agents,
                horizon=horizon,
                objective=a,
                fixed_actions_by_round=None,
                fixed_policy_by_agent=tuple(fixed_policy_by_agent),
                action_candidates_by_agent=action_candidates_by_agent,
                enforce_state_only_for_agents=(a,),
                require_victory=True,
                debug=True,
                timeout_ms=cfg.timeout_ms,
            )
            br_dt = time.perf_counter() - br_t0

            if not br.is_sat:
                if cfg.timing:
                    _log(cfg.progress, f"iter={it} agent={a} smt_br reason={br.reason} time_s={br_dt:.3f}")
                else:
                    _log(cfg.progress, f"iter={it} agent={a} smt_br reason={br.reason}")
                if br.reason == "unknown":
                    return HybridNEResult(False, "unknown", False, it - 1, base_payoff, base, base.actions_by_round)
                continue

            if br.payoff_by_agent is None:
                return HybridNEResult(False, "missing_debug_payoff", False, it - 1, base_payoff, br, br.actions_by_round)

            new_payoff = br.payoff_by_agent[a]
            if cfg.timing:
                _log(cfg.progress, f"iter={it} agent={a} smt_br_payoff={new_payoff} time_s={br_dt:.3f}")
            else:
                _log(cfg.progress, f"iter={it} agent={a} smt_br_payoff={new_payoff}")

            if new_payoff > base_payoff[a]:
                learned = _policy_from_solution_for_agent(br, a)
                delta = int(new_payoff - base_payoff[a])
                ratio = float(new_payoff) / float(base_payoff[a])
                learned_states = len(learned)

                is_better = False
                if best_agent is None or ratio > best_ratio:
                    is_better = True
                elif ratio == best_ratio and delta > best_delta:
                    is_better = True
                elif ratio == best_ratio and delta == best_delta and learned_states < best_learned_states:
                    is_better = True

                if is_better:
                    best_agent = a
                    best_ratio = ratio
                    best_delta = delta
                    best_learned = learned
                    best_learned_states = learned_states

                _log(cfg.progress, f"iter={it} agent={a} improved delta={delta} ratio={ratio:.3f} learned_states={learned_states}")

        if best_agent is None or best_learned is None:
            if cfg.timing:
                _log(cfg.progress, f"iter={it} converged_time_s={time.perf_counter() - iter_t0:.3f}")
                _log(cfg.progress, f"done reason=ne iterations={it} total_time_s={time.perf_counter() - t0:.3f}")
            return HybridNEResult(True, "ne", True, it, base_payoff, base, base.actions_by_round)

        updated = dict(policies[best_agent])
        updated.update(best_learned)
        policies = [dict(p) for p in policies]
        policies[best_agent] = updated

        _log(
            cfg.progress,
            f"iter={it} picked_agent={best_agent} picked_delta={best_delta} picked_ratio={best_ratio:.3f} picked_learned_states={best_learned_states}",
        )
        if cfg.timing:
            _log(cfg.progress, f"iter={it} policy_sizes={[len(p) for p in policies]} iter_time_s={time.perf_counter() - iter_t0:.3f}")
        else:
            _log(cfg.progress, f"iter={it} policy_sizes={[len(p) for p in policies]}")

    final = _evaluate_profile(territory=territory, num_agents=num_agents, horizon=horizon, policies=policies, cfg=cfg)
    if cfg.timing:
        _log(cfg.progress, f"done reason=max_iters iterations={cfg.max_iters} total_time_s={time.perf_counter() - t0:.3f}")

    return HybridNEResult(final.is_sat, "max_iters", False, cfg.max_iters, final.payoff_by_agent, final, final.actions_by_round)


@dataclass(frozen=True, slots=True)
class HybridTestbed(Testbed):
    name: str = "hybrid"

    max_iters: int = 25
    rl_episodes_per_iter: int = 200
    rl_top_k_actions: int = 3
    seed: int = 0

    progress: bool = False
    timing: bool = False
    timeout_ms: int | None = None

    def build_agents(self, *, territory: Territory, num_agents: int, max_rounds: int) -> list[object]:  # noqa: ARG002
        raise NotImplementedError("hybrid does not run via GameEngine; use solve_hybrid_ne")

    def solve(self, *, territory: Territory, num_agents: int, horizon: int) -> HybridNEResult:
        cfg = HybridConfig(
            max_iters=self.max_iters,
            rl_episodes_per_iter=self.rl_episodes_per_iter,
            rl_top_k_actions=self.rl_top_k_actions,
            seed=self.seed,
            progress=self.progress,
            timing=self.timing,
            timeout_ms=self.timeout_ms,
        )
        return solve_hybrid_ne(territory=territory, num_agents=num_agents, horizon=horizon, cfg=cfg)
