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
    """Solve the scenario in one SMT shot.

    This deliberately does *not* interact with the engine loop. We encode the game dynamics
    as constraints and ask Z3 to optimize the collective payoff.
    """

    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")

    try:
        from z3 import And, Bool, If, Implies, Int, Optimize, Or, Sum, is_true, sat
    except Exception as e:  # pragma: no cover
        raise RuntimeError("z3-solver is required for smt-co testbed") from e

    sectors = territory.ordered_sectors()
    S = len(sectors)
    A = num_agents
    T = horizon

    opt = Optimize()

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

    # Objective: maximize total payoff.
    opt.maximize(Sum(payoff))

    if opt.check() != sat:
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
