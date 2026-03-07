"""Algorithm 1 (EUMAS): CEGAR-NE — Abstraction-refinement loop for NE synthesis.

Implements the counterexample-guided abstraction refinement loop:
abstract game → synthesise NE → lift → verify in concrete → refine partition.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from .abstraction import (
    LiftedStrategy,
    Partition,
    build_abstract_rcmas,
    compute_deviation_set,
    discrete_partition,
    lift_strategy,
    orbit_partition,
    refine_partition,
)
from .model import Coord, Territory
from .smt_solve import SmtSolution, solve_smt_game
from .strategy import Policy, policy_from_solution


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CegarResult:
    """Result of the CEGAR-NE loop."""

    is_sat: bool
    reason: str  # "ne" | "no_abstract_ne" | "max_refinements" | "concrete_fallback"
    found_ne: bool
    iterations: int
    final_partition_size: int
    strategy: tuple[tuple[Coord | None, ...], ...] | None
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None


# ---------------------------------------------------------------------------
# NE verification
# ---------------------------------------------------------------------------

def verify_ne(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    lifted_strategy: LiftedStrategy,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    demands: tuple[int, ...] | None = None,
) -> tuple[int | None, SmtSolution | None, tuple[int, ...] | None]:
    """Verify whether a lifted strategy is a Nash equilibrium in the concrete game.

    For each agent, fix the other agents' policies (derived from the lifted
    strategy) and maximise the agent's payoff.  If any agent can improve,
    return the deviating agent, the deviation solution, and the deviated
    terminal ownership.

    Returns ``(None, None, None)`` if the strategy is verified as NE.
    """
    # Build policies from the lifted strategy's action trace
    # We need owner_by_round to build policies.  Simulate from the actions.
    from .model import State, evolve

    state = State.initial(territory, num_agents)
    owner_snapshots: list[tuple[int, ...]] = [state.owner_by_index]
    for t in range(len(lifted_strategy.actions_by_round)):
        step = lifted_strategy.actions_by_round[t]
        actions: dict[int, Coord | None] = {}
        for a in range(num_agents):
            actions[a] = step[a]
        state = evolve(state, actions)
        owner_snapshots.append(state.owner_by_index)

    owner_by_round = tuple(owner_snapshots)
    actions_by_round = lifted_strategy.actions_by_round
    concrete_horizon = len(actions_by_round)

    policies: list[Policy] = []
    for a in range(num_agents):
        pol = policy_from_solution(owner_by_round, actions_by_round, a)
        policies.append(pol)

    # For each agent: fix opponents, maximise agent's payoff
    for a in range(num_agents):
        fixed: list[Policy | None] = [
            None if other == a else policies[other]
            for other in range(num_agents)
        ]

        br = solve_smt_game(
            territory=territory,
            num_agents=num_agents,
            horizon=concrete_horizon,
            objective=a,
            fixed_policy_by_agent=tuple(fixed),
            enforce_state_only_for_agents=(a,),
            require_victory=True,
            timeout_ms=timeout_ms,
            debug=True,
            symmetry_breaking=symmetry,
            demands=demands,
        )

        if not br.is_sat or br.payoff_by_agent is None:
            continue

        if br.payoff_by_agent[a] > lifted_strategy.payoff_by_agent[a]:
            # Agent can improve — this is a counterexample
            assert br.final_state is not None
            return a, br, br.final_state.owner_by_index

    return None, None, None


# ---------------------------------------------------------------------------
# Main CEGAR-NE loop (Algorithm 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _SynthResult:
    """Unified result type for IBIS and Q-IBIS when used by CEGAR."""

    is_sat: bool
    reason: str
    found_ne: bool
    iterations: int
    strategy: tuple[tuple[Coord | None, ...], ...] | None
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None


def _run_synthesiser(
    *,
    synthesiser: str,
    territory: Territory,
    num_agents: int,
    horizon: int,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    demands: tuple[int, ...] | None = None,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
) -> _SynthResult:
    """Dispatch to IBIS or Q-IBIS and return a unified result."""
    if synthesiser == "ibis":
        from .ibis import solve_ibis

        res = solve_ibis(
            territory=territory,
            num_agents=num_agents,
            horizon=horizon,
            progress=progress,
            timing=timing,
            timeout_ms=timeout_ms,
            symmetry=symmetry,
            demands=demands,
            weights=weights,
            custom_neighbors=custom_neighbors,
            weight_balance_target=weight_balance_target,
        )
        return _SynthResult(
            is_sat=res.is_sat,
            reason=res.reason,
            found_ne=res.found_ne,
            iterations=res.iterations,
            strategy=res.strategy,
            payoff_by_agent=res.payoff_by_agent,
            final_solution=res.final_solution,
        )
    elif synthesiser == "qibis":
        from .qibis import QibisConfig, solve_qibis

        cfg = QibisConfig(
            progress=progress,
            timing=timing,
            timeout_ms=timeout_ms,
            symmetry=symmetry,
            demands=demands,
        )
        res = solve_qibis(
            territory=territory,
            num_agents=num_agents,
            horizon=horizon,
            cfg=cfg,
        )
        return _SynthResult(
            is_sat=res.is_sat,
            reason=res.reason,
            found_ne=res.found_ne,
            iterations=res.iterations,
            strategy=res.strategy,
            payoff_by_agent=res.payoff_by_agent,
            final_solution=res.final_solution,
        )
    else:
        raise ValueError(f"unknown synthesiser: {synthesiser!r}")


def solve_cegar(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int,
    initial_partition: str | Partition = "orbit",
    synthesiser: str = "ibis",
    max_iters: int = 25,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    demands: tuple[int, ...] | None = None,
) -> CegarResult:
    """Run the CEGAR-NE loop (Algorithm 1, EUMAS).

    Parameters
    ----------
    territory : Territory
        The concrete territory.
    num_agents : int
        Number of agents.
    horizon : int
        Concrete game horizon (ignored for abstract games which compute
        their own horizon from partition size).
    initial_partition : str or Partition
        ``"orbit"`` (default) starts from symmetry orbits,
        ``"discrete"`` starts fully refined (skips abstraction).
    synthesiser : str
        ``"ibis"`` (default) or ``"qibis"``.
    max_iters : int
        Maximum refinement iterations.
    progress : bool
        Print per-iteration progress to stderr.
    timing : bool
        Include timing information in progress output.
    timeout_ms : int or None
        Z3 timeout per individual solve call.
    symmetry : bool
        Enable symmetry-breaking constraints in SMT calls.
    """

    def _log(msg: str) -> None:
        if progress:
            print(msg, file=sys.stderr, flush=True)

    if synthesiser not in ("ibis", "qibis"):
        raise ValueError("synthesiser must be 'ibis' or 'qibis'")

    # Build initial partition
    if isinstance(initial_partition, str):
        if initial_partition == "orbit":
            partition = orbit_partition(territory)
        elif initial_partition == "discrete":
            partition = discrete_partition(territory)
        else:
            raise ValueError(f"unknown partition type: {initial_partition!r}")
    else:
        partition = initial_partition

    t0 = time.perf_counter()

    for it in range(1, max_iters + 1):
        iter_t0 = time.perf_counter()
        _log(f"cegar iter={it} blocks={len(partition.blocks)}")

        # If partition is fully discrete, fall through to concrete solve
        if len(partition.blocks) >= len(territory):
            _log(f"cegar iter={it} partition=discrete, falling through to concrete {synthesiser}")
            concrete_result = _run_synthesiser(
                synthesiser=synthesiser,
                territory=territory,
                num_agents=num_agents,
                horizon=horizon,
                progress=progress,
                timing=timing,
                timeout_ms=timeout_ms,
                symmetry=symmetry,
                demands=demands,
            )
            if timing:
                _log(f"cegar done reason=concrete_fallback total_time_s={time.perf_counter() - t0:.3f}")
            return CegarResult(
                is_sat=concrete_result.is_sat,
                reason="concrete_fallback",
                found_ne=concrete_result.found_ne,
                iterations=it,
                final_partition_size=len(partition.blocks),
                strategy=concrete_result.strategy,
                payoff_by_agent=concrete_result.payoff_by_agent,
                final_solution=concrete_result.final_solution,
            )

        # 1. Build abstract RCMAS
        abstract = build_abstract_rcmas(territory, num_agents, partition)

        # If the number of blocks is not divisible by num_agents, the abstract
        # game is infeasible (require_victory can't be satisfied).  Refine by
        # splitting the largest block.
        if len(partition.blocks) % num_agents != 0:
            _log(f"cegar iter={it} blocks={len(partition.blocks)} not divisible by {num_agents}, refining largest block")
            largest_idx = max(range(len(partition.blocks)), key=lambda p: len(partition.blocks[p]))
            block = partition.blocks[largest_idx]
            sorted_sectors = sorted(block)
            mid = len(sorted_sectors) // 2
            delta = frozenset(sorted_sectors[:mid])
            partition = refine_partition(partition, delta)
            continue

        # If partition is too coarse (fewer blocks than agents), abstract
        # game has zero horizon — fall through to concrete.
        if abstract.horizon < 1:
            _log(f"cegar iter={it} partition too coarse (blocks={len(partition.blocks)} < agents={num_agents}), falling through to concrete {synthesiser}")
            concrete_result = _run_synthesiser(
                synthesiser=synthesiser,
                territory=territory,
                num_agents=num_agents,
                horizon=horizon,
                progress=progress,
                timing=timing,
                timeout_ms=timeout_ms,
                symmetry=symmetry,
                demands=demands,
            )
            if timing:
                _log(f"cegar done reason=concrete_fallback total_time_s={time.perf_counter() - t0:.3f}")
            return CegarResult(
                is_sat=concrete_result.is_sat,
                reason="concrete_fallback",
                found_ne=concrete_result.found_ne,
                iterations=it,
                final_partition_size=len(partition.blocks),
                strategy=concrete_result.strategy,
                payoff_by_agent=concrete_result.payoff_by_agent,
                final_solution=concrete_result.final_solution,
            )

        # 2. Synthesise NE on abstract game
        concrete_demand = len(territory) // num_agents
        abstract_result = _run_synthesiser(
            synthesiser=synthesiser,
            territory=abstract.territory,
            num_agents=num_agents,
            horizon=abstract.horizon,
            weights=abstract.weights,
            custom_neighbors=abstract.neighbors,
            weight_balance_target=concrete_demand,
            progress=False,
            timeout_ms=timeout_ms,
            symmetry=symmetry,
            demands=demands,
        )

        if timing:
            _log(f"cegar iter={it} abstract_solve_time_s={time.perf_counter() - iter_t0:.3f}")

        # 3. If abstract is unsat, refine by splitting the largest block.
        #    Abstract unsat may be caused by the partition being too coarse
        #    (e.g. no weight-balanced allocation exists), not by absence of
        #    a concrete NE.
        if not abstract_result.is_sat or abstract_result.final_solution is None:
            _log(f"cegar iter={it} abstract=unsat reason={abstract_result.reason}, refining largest block")
            largest_idx = max(range(len(partition.blocks)), key=lambda p: len(partition.blocks[p]))
            block = partition.blocks[largest_idx]
            if len(block) <= 1:
                # Cannot refine further — partition is (nearly) discrete
                _log(f"cegar iter={it} cannot refine further, falling through to concrete")
                continue  # will hit the discrete fallthrough check at top of loop
            sorted_sectors = sorted(block)
            mid = len(sorted_sectors) // 2
            delta = frozenset(sorted_sectors[:mid])
            partition = refine_partition(partition, delta)
            continue

        _log(f"cegar iter={it} abstract_ne={abstract_result.found_ne} abstract_payoff={abstract_result.payoff_by_agent}")

        # 4. Lift abstract solution to concrete
        lifted = lift_strategy(abstract, abstract_result.final_solution)
        _log(f"cegar iter={it} concrete_payoff={lifted.payoff_by_agent}")

        # 5. Verify NE in concrete game
        verify_t0 = time.perf_counter()
        dev_agent, dev_sol, dev_terminal = verify_ne(
            territory=territory,
            num_agents=num_agents,
            horizon=len(lifted.actions_by_round),
            lifted_strategy=lifted,
            timeout_ms=timeout_ms,
            symmetry=symmetry,
            demands=demands,
        )
        if timing:
            _log(f"cegar iter={it} verify_time_s={time.perf_counter() - verify_t0:.3f}")

        # 6. If verified, return NE
        if dev_agent is None:
            _log(f"cegar iter={it} verified=NE")
            if timing:
                _log(f"cegar done reason=ne total_time_s={time.perf_counter() - t0:.3f}")
            return CegarResult(
                is_sat=True,
                reason="ne",
                found_ne=True,
                iterations=it,
                final_partition_size=len(partition.blocks),
                strategy=lifted.actions_by_round,
                payoff_by_agent=lifted.payoff_by_agent,
                final_solution=abstract_result.final_solution,
            )

        # 7. Compute deviation set and refine
        assert dev_terminal is not None
        delta = compute_deviation_set(lifted.terminal_owner, dev_terminal)
        _log(f"cegar iter={it} deviating_agent={dev_agent} deviation_set_size={len(delta)}")

        old_size = len(partition.blocks)
        partition = refine_partition(partition, delta)
        _log(f"cegar iter={it} refined blocks={old_size}->{len(partition.blocks)}")

        if timing:
            _log(f"cegar iter={it} iter_time_s={time.perf_counter() - iter_t0:.3f}")

    # Max iterations reached
    if timing:
        _log(f"cegar done reason=max_refinements total_time_s={time.perf_counter() - t0:.3f}")
    return CegarResult(
        is_sat=False,
        reason="max_refinements",
        found_ne=False,
        iterations=max_iters,
        final_partition_size=len(partition.blocks),
        strategy=None,
        payoff_by_agent=None,
        final_solution=None,
    )
