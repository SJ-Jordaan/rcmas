"""IBIS and S-IBIS (thesis Algorithms 1 and 2, Sections 4.7 and 4.8).

Both run the loop of Algorithm 1 on a single play.  The initial play
``pi^0`` is the play of a model of ``smt(F)`` (with the symmetry-breaking
constraints of Chapter 5 only if symmetry is on).  In iteration ``i`` the
run stops if ``pi^i`` occurred before (outcome ``"n/c"``).  Otherwise every
agent ``a`` gets a best response ``beta_a`` against the canonical completion
of ``pi^i`` (Definition 4.4.2), and its improvement potential is

    IP^i(a) = r_a(pi_{beta_a}) - r_a(pi^i),

with both rewards evaluated on final states (Definition 3.9.2, Remark
4.3.2).  If every potential is 0, the canonical completion of ``pi^i`` is a
demand-constrained Nash equilibrium (Theorem 4.7.3) and the run returns it.
Otherwise the agent with the largest potential, the smallest index on ties,
replaces the play by its best-response play, and all strategies are
completed again from the new play.  The loop runs for ``i = 0, ..., N``.

IBIS computes each best response with the optimisation query
``smt(G_a(pi^i), reward_a)``; S-IBIS uses the feasibility filter and binary
search of Algorithm 2, which give the same potentials (Theorem 4.8.1).
No query contains symmetry-breaking constraints except the initial one.
When Z3 answers a query with unknown, the run stops with outcome ``"unk"``
(Section 7.1).

Outcomes: ``"ne"`` (equilibrium), ``"n/c"`` (play recurs, ``reason="cycle"``,
or the iteration bound is reached, ``reason="max_iters"``), ``"unk"``
(``reason="unknown"``) and ``"uns"`` (``smt(F)`` is unsatisfiable, so no
satisfying strategy exists; ``reason="unsat"``).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable

from .model import Coord, Territory
from .smt_solve import (
    BaseEncoding,
    ExtraConstraints,
    SmtSolution,
    best_response_binsearch_from_base,
    best_response_from_base,
    build_base_encoding,
    find_feasible_from_base,
)
from .strategy import CanonicalCompletion, Play, StateKey, canonical_completion

OUTCOME_BY_REASON = {
    "ne": "ne",
    "cycle": "n/c",
    "max_iters": "n/c",
    "unknown": "unk",
    "unsat": "uns",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IbisIteration:
    """One iteration of the loop of Algorithm 1.

    Attributes:
        index: the iteration ``i``.
        rewards: ``r_a(pi^i)`` for every agent.
        improvement: ``IP^i(a)`` for every agent (empty if the iteration
            stopped before the potentials were computed).
        chosen: ``a*``, or ``None`` if the iteration did not update the play.
        checks: number of solver checks posed in the iteration.
    """

    index: int
    rewards: tuple[int, ...]
    improvement: tuple[int, ...]
    chosen: int | None
    checks: int


@dataclass(frozen=True, slots=True)
class IbisResult:
    """Result of IBIS or S-IBIS.

    Attributes:
        is_sat: a satisfying play was found (``play`` is set).
        reason: ``"ne"``, ``"cycle"``, ``"max_iters"``, ``"unknown"`` or ``"unsat"``.
        found_ne: the canonical completion of ``play`` is an equilibrium.
        iterations: index ``i`` of the last iteration that ran, which is the
            number of play updates before it.  If the bound was reached it is
            ``N``, and ``play`` is the play produced by the update in iteration
            ``N``.
        strategy: the profiles of ``play`` as coordinates.
        payoff_by_agent: ``r_a(play)``, evaluated on the final state.
        final_solution: the SMT model whose play is ``play``.
        play: the final play ``pi^i`` (tool sector indices).
        outcome: ``"ne"``, ``"n/c"``, ``"unk"`` or ``"uns"``.
        trace: one :class:`IbisIteration` per iteration.

    The returned joint strategy is ``completion()``, the canonical
    completion of ``play`` (Definition 4.4.2) under the sector enumeration
    of the territory (Remark 3.1.1).
    """

    is_sat: bool
    reason: str
    found_ne: bool
    iterations: int
    strategy: tuple[tuple[Coord | None, ...], ...] | None
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None
    play: Play | None = None
    outcome: str = ""
    trace: tuple[IbisIteration, ...] = ()

    @property
    def rewards(self) -> tuple[int, ...] | None:
        """``r_a(play)`` (alias of ``payoff_by_agent``)."""
        return self.payoff_by_agent

    def completion(self) -> CanonicalCompletion | None:
        """The canonical completion of the final play, or ``None``."""
        return None if self.play is None else canonical_completion(self.play)


@dataclass(frozen=True, slots=True)
class BestResponse:
    """A best response of one agent against the canonical completion of a play.

    Attributes:
        status: ``"ok"`` or ``"unknown"`` (Z3 answered unknown).
        reward: ``r_a(pi_beta)`` on the final state; the current reward if
            the agent cannot improve and no model is returned.
        play: ``pi_beta``, or ``None`` if no model is returned.
        solution: the SMT model, if any.
        checks: number of solver checks posed.
    """

    status: str
    reward: int
    play: Play | None
    solution: SmtSolution | None
    checks: int


BestResponseFn = Callable[[Play, int, tuple[int, ...]], BestResponse]
"""``(pi, a, rewards of pi) -> best response of a against the completion of pi``."""


# ---------------------------------------------------------------------------
# The loop of Algorithm 1
# ---------------------------------------------------------------------------


def run_ibis_loop(
    *,
    initial_play: Play,
    initial_solution: SmtSolution | None,
    best_response: BestResponseFn,
    rewards_of: Callable[[StateKey], tuple[int, ...]],
    max_iters: int,
    territory: Territory | None = None,
    log: Callable[[str], None] | None = None,
) -> IbisResult:
    """Run lines 5-20 of Algorithm 1 from *initial_play*.

    *best_response* computes ``beta_a`` for an agent; *rewards_of* evaluates
    ``r_a`` on a final control vector.  The loop runs for ``i = 0, ...,
    max_iters`` (``max_iters`` is the bound ``N``).  *territory* is used
    only to report the play as coordinates.
    """

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    def _result(
        reason: str, play: Play, sol: SmtSolution | None, iterations: int
    ) -> IbisResult:
        coords = play.to_coords(territory) if territory is not None else None
        return IbisResult(
            is_sat=True,
            reason=reason,
            found_ne=reason == "ne",
            iterations=iterations,
            strategy=coords,
            payoff_by_agent=rewards_of(play.final_state),
            final_solution=sol,
            play=play,
            outcome=OUTCOME_BY_REASON[reason],
            trace=tuple(trace),
        )

    if max_iters < 0:
        raise ValueError("max_iters must be >= 0")

    play = initial_play
    solution = initial_solution
    seen: set[Play] = set()
    trace: list[IbisIteration] = []

    for i in range(max_iters + 1):
        current = rewards_of(play.final_state)
        if play in seen:
            _log(f"iter={i} play recurs")
            trace.append(IbisIteration(i, current, (), None, 0))
            return _result("cycle", play, solution, i)
        seen.add(play)

        potentials: list[int] = []
        responses: list[BestResponse] = []
        checks = 0
        for a in range(play.num_agents):
            br = best_response(play, a, current)
            checks += br.checks
            if br.status == "unknown":
                _log(f"iter={i} agent={a} best response: solver answered unknown")
                trace.append(IbisIteration(i, current, tuple(potentials), None, checks))
                return _result("unknown", play, solution, i)
            ip = br.reward - current[a]
            if ip < 0:
                raise RuntimeError(
                    f"negative improvement potential for agent {a} (Lemma 4.7.2(3) violated)"
                )
            potentials.append(ip)
            responses.append(br)
            _log(
                f"iter={i} agent={a} reward={current[a]} best_response={br.reward} ip={ip}"
            )

        if all(ip == 0 for ip in potentials):
            trace.append(IbisIteration(i, current, tuple(potentials), None, checks))
            _log(f"iter={i} rewards={current} equilibrium")
            return _result("ne", play, solution, i)

        chosen = max(range(len(potentials)), key=lambda a: (potentials[a], -a))
        trace.append(IbisIteration(i, current, tuple(potentials), chosen, checks))
        new_play = responses[chosen].play
        assert new_play is not None
        _log(f"iter={i} rewards={current} ip={tuple(potentials)} update agent={chosen}")
        play = new_play
        solution = responses[chosen].solution

    return _result("max_iters", play, solution, max_iters)


# ---------------------------------------------------------------------------
# Best responses over the SMT encoding
# ---------------------------------------------------------------------------


def smt_best_response(
    base: BaseEncoding, timeout_ms: int | None = None
) -> BestResponseFn:
    """IBIS best response: the optimisation query ``smt(G_a(pi), reward_a)`` (line 9)."""

    def br(play: Play, agent: int, current: tuple[int, ...]) -> BestResponse:
        sol = best_response_from_base(base, play, agent, timeout_ms=timeout_ms)
        if sol.reason == "unknown":
            return BestResponse("unknown", current[agent], None, None, 1)
        if not sol.is_sat:
            raise RuntimeError(
                "best-response query unsatisfiable for a satisfying play (Lemma 4.7.2(3) violated)"
            )
        assert sol.play is not None and sol.payoff_by_agent is not None
        return BestResponse("ok", sol.payoff_by_agent[agent], sol.play, sol, 1)

    return br


def binsearch_best_response(
    base: BaseEncoding, timeout_ms: int | None = None
) -> BestResponseFn:
    """S-IBIS best response: feasibility filter and binary search (lines 11-24 of Algorithm 2)."""

    def br(play: Play, agent: int, current: tuple[int, ...]) -> BestResponse:
        res = best_response_binsearch_from_base(
            base, play, agent, current_reward=current[agent], timeout_ms=timeout_ms
        )
        if res.reason == "unknown":
            return BestResponse("unknown", current[agent], None, None, res.checks)
        if res.reason == "no_improvement":
            return BestResponse("ok", current[agent], None, None, res.checks)
        sol = res.solution
        assert (
            sol is not None and sol.play is not None and sol.payoff_by_agent is not None
        )
        return BestResponse("ok", sol.payoff_by_agent[agent], sol.play, sol, res.checks)

    return br


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _validate(
    territory: Territory, num_agents: int, horizon: int | None, max_iters: int
) -> int:
    if num_agents <= 0:
        raise ValueError("num_agents must be >= 1")
    m = len(territory) // num_agents
    if horizon is None:
        horizon = m
    if horizon != m:
        raise ValueError(
            f"horizon must be m = floor(|T|/|Agt|) = {m} (Definition 3.9.1), got {horizon}"
        )
    if max_iters < 0:
        raise ValueError("max_iters must be >= 0")
    return horizon


def _ibis_core(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int | None,
    max_iters: int,
    progress: bool,
    timing: bool,
    timeout_ms: int | None,
    symmetry: bool,
    weights: tuple[int, ...] | None,
    custom_neighbors: dict[int, list[int]] | None,
    weight_balance_target: int | None,
    demands: tuple[int, ...] | None,
    initial_constraints: ExtraConstraints | None,
    initial_play: Play | None,
    make_best_response: Callable[[BaseEncoding, int | None], BestResponseFn],
) -> IbisResult:
    h = _validate(territory, num_agents, horizon, max_iters)
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if progress:
            if timing:
                msg = f"{msg} t={time.perf_counter() - t0:.3f}s"
            print(msg, file=sys.stderr, flush=True)

    base = build_base_encoding(
        territory=territory,
        num_agents=num_agents,
        horizon=h,
        demands=demands,
        weights=weights,
        custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target,
    )

    if initial_play is None:
        abstract = weights is not None or custom_neighbors is not None
        use_symmetry = symmetry and not abstract
        if symmetry and abstract and initial_constraints is None:
            _log(
                "symmetry: no grid symmetry for an abstract game; pass initial_constraints"
            )
        init = find_feasible_from_base(
            base,
            symmetry=use_symmetry,
            extra_constraints=initial_constraints,
            timeout_ms=timeout_ms,
        )
        if init.reason == "unknown":
            _log("initial query: solver answered unknown")
            return IbisResult(
                False, "unknown", False, 0, None, None, None, outcome="unk"
            )
        if not init.is_sat:
            _log("initial query unsatisfiable: no satisfying strategy exists")
            return IbisResult(False, "unsat", False, 0, None, None, None, outcome="uns")
        assert init.play is not None
        start, start_sol = init.play, init
    else:
        if initial_play.num_agents != num_agents or initial_play.horizon != h:
            raise ValueError("initial_play does not match the game")
        start, start_sol = initial_play, None

    return run_ibis_loop(
        initial_play=start,
        initial_solution=start_sol,
        best_response=make_best_response(base, timeout_ms),
        rewards_of=base.rewards_of,
        max_iters=max_iters,
        territory=territory,
        log=_log if progress else None,
    )


def solve_ibis(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int | None = None,
    seed: int = 0,
    max_iters: int = 30,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    demands: tuple[int, ...] | None = None,
    require_victory: bool = False,
    initial_constraints: ExtraConstraints | None = None,
    initial_play: Play | None = None,
) -> IbisResult:
    """Run IBIS (Algorithm 1) with the iteration bound ``N = max_iters``.

    *horizon* must be ``m = floor(|T|/|Agt|)`` (``None`` selects it).
    *symmetry* adds the symmetry-breaking constraints of the territory to the
    initial query only; for an abstract game (*weights* or
    *custom_neighbors* given) pass its constraints as *initial_constraints*.
    *initial_play* replaces the initial query.  *seed* and *require_victory*
    are accepted for backward compatibility and have no effect.
    """
    _ = (seed, require_victory)
    return _ibis_core(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        max_iters=max_iters,
        progress=progress,
        timing=timing,
        timeout_ms=timeout_ms,
        symmetry=symmetry,
        weights=weights,
        custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target,
        demands=demands,
        initial_constraints=initial_constraints,
        initial_play=initial_play,
        make_best_response=smt_best_response,
    )


def solve_sibis(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int | None = None,
    seed: int = 0,
    max_iters: int = 30,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    demands: tuple[int, ...] | None = None,
    require_victory: bool = False,
    initial_constraints: ExtraConstraints | None = None,
    initial_play: Play | None = None,
) -> IbisResult:
    """Run S-IBIS (Algorithm 2): the loop of IBIS with binary-search best responses.

    ``R_max`` is ``m``, or the largest total weight of ``m`` sectors in a
    weighted (abstract) game.  The parameters are those of :func:`solve_ibis`.
    """
    _ = (seed, require_victory)
    return _ibis_core(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        max_iters=max_iters,
        progress=progress,
        timing=timing,
        timeout_ms=timeout_ms,
        symmetry=symmetry,
        weights=weights,
        custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target,
        demands=demands,
        initial_constraints=initial_constraints,
        initial_play=initial_play,
        make_best_response=binsearch_best_response,
    )
