"""IBIS and S-IBIS (thesis Algorithms 1 and 2, Sections 4.7 and 4.8).

Both run the loop of Algorithm 1 on a single play.  The initial play
``pi^0`` is the play of a model of ``smt(F)`` or, with
``initial="collective"``, of the collectively optimal query ``smt(F, f)``
(Definition 4.6.1); with symmetry on, the symmetry-breaking constraints of
Chapter 5 are added to this query and to no other.  A model of either query
is a model of ``F``, which is all that Theorem 4.7.3 uses of ``pi^0``.  In
iteration ``i`` the run stops if ``pi^i`` occurred before (outcome
``"n/c"``).  Otherwise every agent ``a`` gets a best response ``beta_a``
against the canonical completion of ``pi^i`` (Definition 4.4.2), and its
improvement potential is

    IP^i(a) = r_a(pi_{beta_a}) - r_a(pi^i),

with both rewards evaluated on final states (Definition 3.9.2, Remark
4.3.2).  If every potential is 0, the canonical completion of ``pi^i`` is a
demand-constrained Nash equilibrium (Theorem 4.7.3) and the run returns it.
Otherwise the agent with the largest potential, the smallest index on ties,
replaces the play by its best-response play, and all strategies are
completed again from the new play.  The loop runs for ``i = 0, ..., N``, or
without a bound if ``N`` is ``None``.  An unbounded loop also ends: a game
has finitely many plays, and every iteration either returns or moves to a
play that has not occurred before.

IBIS computes each best response with the optimisation query
``smt(G_a(pi^i), reward_a)``; S-IBIS uses the feasibility filter and binary
search of Algorithm 2, which give the same potentials (Theorem 4.8.1).
Neither poses a query for an agent whose reward is already ``R_max``
(``m``, or the largest total weight of ``m`` sectors in an abstract game):
``F`` implies ``reward_a <= R_max``, so its potential is 0
(:func:`at_reward_bound`).  This changes only the number of queries; from
an initial play in which every agent has reward ``m`` a run returns in
iteration 0 without a best-response query.
When Z3 answers a query with unknown, the initial query included, the run
stops with outcome ``"unk"`` (Section 7.1).

Outcomes: ``"ne"`` (equilibrium), ``"n/c"`` (play recurs, ``reason="cycle"``,
or the iteration bound is reached, ``reason="max_iters"``, which cannot
happen without a bound), ``"unk"`` (``reason="unknown"``) and ``"uns"``
(the initial query is unsatisfiable, so no satisfying strategy exists;
``reason="unsat"``).
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable, TypeVar

from .model import Coord, Territory
from .smt_solve import (
    INITIAL_QUERIES,
    BaseEncoding,
    ExtraConstraints,
    SmtSolution,
    best_response_binsearch_from_base,
    best_response_from_base,
    build_base_encoding,
    initial_from_base,
)
from .strategy import CanonicalCompletion, Play, StateKey, canonical_completion

ResultT = TypeVar("ResultT")

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
        initial: the initial query, ``"satisfying"`` or ``"collective"``
            (``"given"`` if the caller passed the initial play).
        initial_query_s: wall-clock time of the initial query (``None`` for
            a given initial play).
        initial_play: ``pi^0``, if the initial query had a model.
        initial_rewards: ``r_a(pi^0)``, evaluated on the final state.

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
    initial: str = "satisfying"
    initial_query_s: float | None = None
    initial_play: Play | None = None
    initial_rewards: tuple[int, ...] | None = None

    @property
    def rewards(self) -> tuple[int, ...] | None:
        """``r_a(play)`` (alias of ``payoff_by_agent``)."""
        return self.payoff_by_agent

    @property
    def returned_initial(self) -> bool:
        """The run returned the initial play ``pi^0`` (an equilibrium in iteration 0)."""
        return self.found_ne and self.iterations == 0

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
    max_iters: int | None,
    territory: Territory | None = None,
    log: Callable[[str], None] | None = None,
) -> IbisResult:
    """Run lines 5-20 of Algorithm 1 from *initial_play*.

    *best_response* computes ``beta_a`` for an agent; *rewards_of* evaluates
    ``r_a`` on a final control vector.  The loop runs for ``i = 0, ...,
    max_iters`` (``max_iters`` is the bound ``N``), or until it returns if
    *max_iters* is ``None``.  *territory* is used only to report the play
    as coordinates.
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

    play = initial_play
    solution = initial_solution
    seen: set[Play] = set()
    trace: list[IbisIteration] = []

    for i in iteration_indices(max_iters):
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

    assert max_iters is not None  # an unbounded loop returns inside
    return _result("max_iters", play, solution, max_iters)


def iteration_indices(max_iters: int | None) -> Iterable[int]:
    """``0, ..., N`` for the bound ``N = max_iters``, or ``0, 1, ...`` without a bound.

    Raises :class:`ValueError` for a negative bound.  A loop over the
    unbounded range ends because it stops at the first recurring play.
    """
    if max_iters is None:
        return itertools.count()
    if max_iters < 0:
        raise ValueError("max_iters must be >= 0 or None")
    return range(max_iters + 1)


# ---------------------------------------------------------------------------
# Best responses over the SMT encoding
# ---------------------------------------------------------------------------


def at_reward_bound(base: BaseEncoding, current: tuple[int, ...], agent: int) -> bool:
    """``r_a(pi) = R_max``: *agent* cannot improve, so no query is needed.

    ``F`` implies ``reward_a <= R_max`` (every agent controls exactly ``m``
    sectors of a collision-free final state, Lemma 3.10.1, and the witness
    region lies among them, Lemma 4.3.1), so ``BR_a(pi) <= R_max`` and the
    improvement potential of an agent at the bound is 0.  An agent with
    potential 0 is never the one updated, so skipping its query leaves the
    run unchanged apart from the number of queries.
    """
    return current[agent] >= base.max_reward


def smt_best_response(
    base: BaseEncoding, timeout_ms: int | None = None
) -> BestResponseFn:
    """IBIS best response: the optimisation query ``smt(G_a(pi), reward_a)`` (line 9)."""

    def br(play: Play, agent: int, current: tuple[int, ...]) -> BestResponse:
        if at_reward_bound(base, current, agent):
            return BestResponse("ok", current[agent], None, None, 0)
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
        if at_reward_bound(base, current, agent):
            return BestResponse("ok", current[agent], None, None, 0)
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
    territory: Territory,
    num_agents: int,
    horizon: int | None,
    max_iters: int | None,
    initial: str = "satisfying",
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
    if max_iters is not None and max_iters < 0:
        raise ValueError("max_iters must be >= 0 or None")
    if initial not in INITIAL_QUERIES:
        raise ValueError(f"initial must be one of {INITIAL_QUERIES}, got {initial!r}")
    return horizon


@dataclass(frozen=True, slots=True)
class InitialPlay:
    """The outcome of the initial query of a run (lines 1-4 of Algorithm 1).

    Attributes:
        reason: ``"sat"``, ``"unsat"``, ``"unknown"`` or ``"given"`` (the
            caller passed the initial play).
        play: ``pi^0``, if any.
        solution: the model whose play is ``pi^0``, if any.
        rewards: ``r_a(pi^0)``, if any.
        query_s: wall-clock time of the query (``None`` for a given play).
        initial: ``"satisfying"``, ``"collective"`` or ``"given"``.
    """

    reason: str
    play: Play | None
    solution: SmtSolution | None
    rewards: tuple[int, ...] | None
    query_s: float | None
    initial: str


InitialHook = Callable[[InitialPlay], None]
"""Called with the outcome of the initial query as soon as it is known."""


def initial_play_of(
    base: BaseEncoding,
    *,
    initial: str,
    symmetry: bool,
    extra_constraints: ExtraConstraints | None,
    timeout_ms: int | None,
    given: Play | None = None,
) -> InitialPlay:
    """Pose the initial query of IBIS, S-IBIS or Q-IBIS, or take the *given* play.

    *initial* selects ``smt(F)`` (``"satisfying"``) or ``smt(F, f)``
    (``"collective"``, Definition 4.6.1); *symmetry* adds the
    symmetry-breaking constraints of the territory and *extra_constraints*
    further constraints, both to this query only.
    """
    if given is not None:
        if given.num_agents != base.num_agents or given.horizon != base.horizon:
            raise ValueError("initial_play does not match the game")
        return InitialPlay(
            "given", given, None, base.rewards_of(given.final_state), None, "given"
        )
    t0 = time.perf_counter()
    sol = initial_from_base(
        base,
        initial=initial,
        symmetry=symmetry,
        extra_constraints=extra_constraints,
        timeout_ms=timeout_ms,
    )
    elapsed = time.perf_counter() - t0
    if not sol.is_sat:
        return InitialPlay(sol.reason, None, None, None, elapsed, initial)
    assert sol.play is not None and sol.payoff_by_agent is not None
    return InitialPlay("sat", sol.play, sol, sol.payoff_by_agent, elapsed, initial)


def _ibis_core(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int | None,
    max_iters: int | None,
    initial: str,
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
    on_initial: InitialHook | None = None,
) -> IbisResult:
    h = _validate(territory, num_agents, horizon, max_iters, initial)
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

    abstract = weights is not None or custom_neighbors is not None
    if initial_play is None and symmetry and abstract and initial_constraints is None:
        _log(
            "symmetry: no grid symmetry for an abstract game; pass initial_constraints"
        )
    init = initial_play_of(
        base,
        initial=initial,
        symmetry=symmetry and not abstract,
        extra_constraints=initial_constraints,
        timeout_ms=timeout_ms,
        given=initial_play,
    )
    if on_initial is not None:
        on_initial(init)
    if init.reason == "unknown":
        _log(f"initial query ({initial}): solver answered unknown")
        return failed_initial(IbisResult, init)
    if init.play is None:
        _log("initial query unsatisfiable: no satisfying strategy exists")
        return failed_initial(IbisResult, init)
    if init.query_s is not None:
        _log(
            f"initial query ({initial}): rewards={init.rewards} in {init.query_s:.3f}s"
        )

    res = run_ibis_loop(
        initial_play=init.play,
        initial_solution=init.solution,
        best_response=make_best_response(base, timeout_ms),
        rewards_of=base.rewards_of,
        max_iters=max_iters,
        territory=territory,
        log=_log if progress else None,
    )
    return with_initial(res, init)


def failed_initial(cls: Callable[..., ResultT], init: InitialPlay) -> ResultT:
    """The result of a run whose initial query had no model (``unk`` or ``uns``)."""
    reason = "unknown" if init.reason == "unknown" else "unsat"
    return cls(
        False,
        reason,
        False,
        0,
        None,
        None,
        None,
        outcome=OUTCOME_BY_REASON[reason],
        initial=init.initial,
        initial_query_s=init.query_s,
    )


def with_initial(res: ResultT, init: InitialPlay) -> ResultT:
    """*res* with the fields that describe its initial play."""
    return replace(  # type: ignore[type-var]
        res,
        initial=init.initial,
        initial_query_s=init.query_s,
        initial_play=init.play,
        initial_rewards=init.rewards,
    )


def solve_ibis(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int | None = None,
    seed: int = 0,
    max_iters: int | None = 30,
    initial: str = "satisfying",
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
    on_initial: InitialHook | None = None,
) -> IbisResult:
    """Run IBIS (Algorithm 1) with the iteration bound ``N = max_iters``.

    *max_iters* ``None`` runs the loop without a bound: it stops at an
    equilibrium or at the first recurring play.  *initial* selects the
    query for the initial play, ``"satisfying"`` (``smt(F)``, the default)
    or ``"collective"`` (``smt(F, f)``, Definition 4.6.1).  *horizon* must
    be ``m = floor(|T|/|Agt|)`` (``None`` selects it).  *symmetry* adds the
    symmetry-breaking constraints of the territory to the initial query
    only; for an abstract game (*weights* or *custom_neighbors* given) pass
    its constraints as *initial_constraints*.  *initial_play* replaces the
    initial query.  *on_initial* is called with the outcome of the initial
    query (an :class:`InitialPlay`) as soon as it is known.  *seed* and
    *require_victory* are accepted for backward compatibility and have no
    effect.
    """
    _ = (seed, require_victory)
    return _ibis_core(
        territory=territory,
        num_agents=num_agents,
        horizon=horizon,
        max_iters=max_iters,
        initial=initial,
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
        on_initial=on_initial,
    )


def solve_sibis(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int | None = None,
    seed: int = 0,
    max_iters: int | None = 30,
    initial: str = "satisfying",
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
    on_initial: InitialHook | None = None,
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
        initial=initial,
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
        on_initial=on_initial,
    )
