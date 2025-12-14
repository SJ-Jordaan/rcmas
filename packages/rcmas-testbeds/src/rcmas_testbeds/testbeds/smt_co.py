from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from rcmas_core.engine import Coord, GameState, Territory

from .base import Testbed


@dataclass(frozen=True, slots=True)
class SmtSolution:
    is_sat: bool
    reason: str
    final_state: GameState | None
    actions_by_round: tuple[tuple[Coord | None, ...], ...] | None
    # Debug/inspection data (only populated when debug=True)
    owner_by_round: tuple[tuple[int, ...], ...] | None = None
    payoff_by_agent: tuple[int, ...] | None = None
    size_by_seed: tuple[tuple[int, ...], ...] | None = None
    best_seed_by_agent: tuple[int | None, ...] | None = None
    best_region_by_agent: tuple[tuple[Coord, ...] | None, ...] | None = None


def solve_collective_optimality(*, territory: Territory, num_agents: int, horizon: int, debug: bool = False) -> SmtSolution:
    """Solve the scenario in one SMT shot (collective optimality).

    This deliberately does *not* interact with the engine loop. We encode the game dynamics
    as constraints and ask Z3 to optimize the collective payoff.
    """

    return solve_smt_game(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        objective="sum",
        fixed_actions_by_round=None,
        require_victory=False,
        debug=debug,
    )


def solve_smt_game(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    objective: str | int = "sum",
    fixed_actions_by_round: tuple[tuple[Coord | None, ...], ...] | None = None,
    fixed_policy_by_agent: tuple[dict[tuple[int, ...], Coord | None] | None, ...] | None = None,
    enforce_state_only_for_agents: tuple[int, ...] = (),
    require_victory: bool = False,
    debug: bool = False,
    timeout_ms: int | None = None,
) -> SmtSolution:
    """Generic SMT solve for this game's dynamics.

    See module docstring for details.
    """

    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")

    try:
        from z3 import And, Bool, If, Implies, Int, Optimize, Or, Sum, is_true, sat, unknown
    except Exception as e:  # pragma: no cover
        raise RuntimeError("z3-solver is required for smt-co testbed") from e

    sectors = territory.ordered_sectors()
    S = len(sectors)
    A = num_agents
    T = horizon

    opt = Optimize()
    if timeout_ms is not None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        opt.set(timeout=timeout_ms)

    # owner[s][t] in {-1 (unowned), 0..A-1}
    owner = [[Int(f"owner_{s}_{t}") for t in range(T + 1)] for s in range(S)]
    for s in range(S):
        for t in range(T + 1):
            opt.add(And(owner[s][t] >= -1, owner[s][t] < A))

    # action[a][t] in {-1 (no-op), 0..S-1}
    action = [[Int(f"action_{a}_{t}") for t in range(T)] for a in range(A)]
    for a in range(A):
        for t in range(T):
            opt.add(Or(action[a][t] == -1, And(action[a][t] >= 0, action[a][t] < S)))

    # Optionally fix actions for some or all agents per round.
    if fixed_actions_by_round is not None:
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
                opt.add(action[a][t] == sidx)

    # Optionally fix actions via per-agent state->action policies.
    #
    # Each fixed agent's action at time t is constrained to be a pure function of the
    # current ownership vector owner[*][t]. Policies may be partial; any unmatched
    # state uses a deterministic fallback action computed from the state.
    if fixed_policy_by_agent is not None:
        if len(fixed_policy_by_agent) != A:
            raise ValueError("fixed_policy_by_agent must have length == num_agents")

        # Validate indices for any requested state-only consistency constraints.
        for a in enforce_state_only_for_agents:
            if a < 0 or a >= A:
                raise ValueError("enforce_state_only_for_agents contains invalid agent index")

        def default_action_expr(agent_id: int, t: int):
            # Deterministic collision-free fallback: choose the agent_id-th unowned
            # sector in the fixed ordering (else -1 if fewer unowned remain).
            expr = -1
            for i in reversed(range(S)):
                unowned_before = Sum([If(owner[j][t] == -1, 1, 0) for j in range(i)])
                cond = And(owner[i][t] == -1, unowned_before == agent_id)
                expr = If(cond, i, expr)
            return expr

        # Apply policy constraints.
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

        # Constrain action[a][t] for fixed-policy agents.
        for t in range(T):
            for a in range(A):
                policy = fixed_policy_by_agent[a]
                if policy is None:
                    continue

                expr = default_action_expr(a, t)

                # Stable iteration order for reproducibility.
                for state in sorted(policy.keys()):
                    chosen_coord = policy[state]
                    chosen_idx = -1
                    if chosen_coord is not None:
                        sidx = territory.index_of(chosen_coord)
                        if sidx is None:
                            raise ValueError(f"fixed policy action {chosen_coord} is not in territory")
                        chosen_idx = sidx

                    match_state = And([owner[s][t] == state[s] for s in range(S)])
                    expr = If(match_state, chosen_idx, expr)

                opt.add(action[a][t] == expr)

        # Optional: enforce that selected agents are state-only (same state => same action)
        # across the whole horizon. Useful when synthesizing / extracting state-only policies.
        for a in enforce_state_only_for_agents:
            for t1 in range(T):
                for t2 in range(t1 + 1, T):
                    same_state = And([owner[s][t1] == owner[s][t2] for s in range(S)])
                    opt.add(Implies(same_state, action[a][t1] == action[a][t2]))

    # Initial state: everything unowned.
    for s in range(S):
        opt.add(owner[s][0] == -1)

    # Action availability + no overwriting: can only claim unowned at time t.
    for t in range(T):
        for a in range(A):
            for s in range(S):
                opt.add(Implies(action[a][t] == s, owner[s][t] == -1))

    # No collisions at each timestep (ignoring -1).
    for t in range(T):
        for a1 in range(A):
            for a2 in range(a1 + 1, A):
                opt.add(
                    If(
                        action[a1][t] >= 0,
                        If(action[a2][t] >= 0, action[a1][t] != action[a2][t], True),
                        True,
                    )
                )

    # Evolution: owner[s][t+1] updates if some agent claimed s at t.
    for t in range(T):
        for s in range(S):
            # claim_sum is 0 if nobody claimed, else (agent_id+1) since at most one.
            claim_sum = Sum([If(action[a][t] == s, a + 1, 0) for a in range(A)])
            opt.add(owner[s][t + 1] == If(claim_sum == 0, owner[s][t], claim_sum - 1))

    # Optional: enforce victory (all sectors claimed) at the final timestep.
    if require_victory:
        for s in range(S):
            opt.add(owner[s][T] != -1)

    # --- Payoff: largest connected component per agent at final timestep ---
    # We mirror the old rcmas idea: build adjacency at final_t, compute reachability,
    # then payoff[a] is max size over all seeds.
    final_t = T

    # Precompute physical neighbors in the territory graph.
    neighbors: dict[int, list[int]] = {i: [] for i in range(S)}
    index_by_coord = {c: i for i, c in enumerate(sectors)}
    for i, c in enumerate(sectors):
        for nb in (Coord(c.x + 1, c.y), Coord(c.x - 1, c.y), Coord(c.x, c.y + 1), Coord(c.x, c.y - 1)):
            j = index_by_coord.get(nb)
            if j is not None:
                neighbors[i].append(j)

    # Define adj(i,j,a) for all pairs i<j, but force it true only for physical neighbors
    # owned by agent a at the final timestep.
    adj: dict[tuple[int, int, int], Any] = {}
    for a in range(A):
        for i in range(S):
            for j in range(i + 1, S):
                adj[(i, j, a)] = Bool(f"adj_{i}_{j}_{a}")

    def _is_phys_adj(i: int, j: int) -> bool:
        return j in neighbors[i]

    def get_adj(i: int, j: int, a: int):
        lo, hi = (i, j) if i < j else (j, i)
        return adj[(lo, hi, a)]

    for a in range(A):
        for i in range(S):
            for j in range(i + 1, S):
                opt.add(get_adj(i, j, a) == And(_is_phys_adj(i, j), owner[i][final_t] == a, owner[j][final_t] == a))

    # cr(i,j,a) for all pairs i<j.
    cr: dict[tuple[int, int, int], Any] = {}
    for a in range(A):
        for i in range(S):
            for j in range(i + 1, S):
                cr[(i, j, a)] = Bool(f"cr_{i}_{j}_{a}")

    def get_cr(i: int, j: int, a: int):
        lo, hi = (i, j) if i < j else (j, i)
        return cr[(lo, hi, a)]

    # Ordered recursion over interval length to avoid cyclic definitions.
    for a in range(A):
        for span in range(1, S):
            for i in range(S - span):
                j = i + span
                chain_terms = []
                for k in range(i + 1, j):
                    chain_terms.append(And(get_adj(i, k, a), get_cr(k, j, a)))
                    chain_terms.append(And(get_cr(i, k, a), get_adj(k, j, a)))
                opt.add(get_cr(i, j, a) == Or([get_adj(i, j, a)] + chain_terms))

    # size[i][a], payoff[a]
    size = [[Int(f"size_{i}_{a}") for a in range(A)] for i in range(S)]
    payoff = [Int(f"payoff_{a}") for a in range(A)]

    for i in range(S):
        for a in range(A):
            connections = Sum([If(get_cr(i, j, a), 1, 0) for j in range(i + 1, S)])
            opt.add(size[i][a] == If(owner[i][final_t] == a, 1 + connections, 0))

    for a in range(A):
        opt.add(payoff[a] >= 0)
        for i in range(S):
            opt.add(payoff[a] >= size[i][a])
        opt.add(Or([payoff[a] == size[i][a] for i in range(S)]))

    # Objective
    if objective == "sum":
        opt.maximize(Sum(payoff))
    elif isinstance(objective, int):
        if objective < 0 or objective >= A:
            raise ValueError("objective agent index out of range")
        opt.maximize(payoff[objective])
    else:
        raise ValueError("objective must be 'sum' or an agent index")

    check_res = opt.check()
    if check_res != sat:
        if check_res == unknown:
            return SmtSolution(False, "unknown", None, None)
        return SmtSolution(False, "unsat", None, None)

    model = opt.model()

    # Extract full action plan and final ownership.
    actions_by_round: list[tuple[Coord | None, ...]] = []
    for t in range(T):
        step: list[Coord | None] = []
        for a in range(A):
            chosen_ref = cast(Any, model.eval(action[a][t], model_completion=True))
            chosen = int(chosen_ref.as_long())
            step.append(None if chosen < 0 else sectors[chosen])
        actions_by_round.append(tuple(step))

    final_owner_by_index: list[int] = []
    for s in range(S):
        own_ref = cast(Any, model.eval(owner[s][final_t], model_completion=True))
        final_owner_by_index.append(int(own_ref.as_long()))

    final_state = GameState(
        territory=territory,
        num_agents=A,
        owner_by_index=tuple(final_owner_by_index),
        round_index=T,
    )

    if not debug:
        return SmtSolution(True, "sat", final_state, tuple(actions_by_round))

    # --- Debug extraction: raw owner history + payoff/size vars + a concrete cohesive region witness ---
    owner_by_round: list[tuple[int, ...]] = []
    for t in range(T + 1):
        snapshot: list[int] = []
        for s in range(S):
            snap_ref = cast(Any, model.eval(owner[s][t], model_completion=True))
            snapshot.append(int(snap_ref.as_long()))
        owner_by_round.append(tuple(snapshot))

    payoff_by_agent: list[int] = []
    for a in range(A):
        p_ref = cast(Any, model.eval(payoff[a], model_completion=True))
        payoff_by_agent.append(int(p_ref.as_long()))

    size_by_seed: list[tuple[int, ...]] = []
    for i in range(S):
        row: list[int] = []
        for a in range(A):
            s_ref = cast(Any, model.eval(size[i][a], model_completion=True))
            row.append(int(s_ref.as_long()))
        size_by_seed.append(tuple(row))

    # Find best seed per agent and list the corresponding region coords.
    best_seed_by_agent: list[int | None] = []
    best_region_by_agent: list[tuple[Coord, ...] | None] = []

    # Evaluate cr for membership queries (only needs final_t).
    def cr_true(i: int, j: int, a: int) -> bool:
        if i == j:
            return True
        lo, hi = (i, j) if i < j else (j, i)
        v = get_cr(lo, hi, a)
        b_ref = cast(Any, model.eval(v, model_completion=True))
        return is_true(b_ref)

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

        # Witness set: all sectors connected (per cr) to the chosen seed.
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


@dataclass(frozen=True, slots=True)
class SmtCollectiveOptimalityTestbed(Testbed):
    name: str = "smt-co"

    def build_agents(self, *, territory: Territory, num_agents: int, max_rounds: int) -> list[object]:  # noqa: ARG002
        raise NotImplementedError("smt-co does not run via GameEngine; use solve_collective_optimality")

    def solve(self, *, territory: Territory, num_agents: int, horizon: int) -> SmtSolution:
        return solve_collective_optimality(territory=territory, num_agents=num_agents, horizon=horizon)

    def solve_debug(self, *, territory: Territory, num_agents: int, horizon: int) -> SmtSolution:
        return solve_collective_optimality(territory=territory, num_agents=num_agents, horizon=horizon, debug=True)
