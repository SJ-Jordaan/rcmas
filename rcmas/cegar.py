"""CEGAR-NE: counterexample-guided NE synthesis (thesis Algorithm 4, Chapter 6).

:func:`solve_cegar` runs the loop of Algorithm 4 from an initial partition
``P_0`` of the territory (the orbit partition by default, Section 6.5):

1. (line 5) build the abstract RCMAS of ``P`` (Definition 6.1.2), with block
   weights and block adjacency;
2. (line 6) run the inner synthesiser, S-IBIS by default, or IBIS or Q-IBIS,
   on it with the same iteration bound ``N`` as a direct run.  At the
   discrete partition the abstract game is ``M`` (Section 6.1), and the
   synthesiser runs on the concrete encoding of ``M``;
3. (lines 7-12) if it returns no equilibrium (the abstract game has no
   satisfying strategy, or the synthesiser stops with ``n/c``), return
   FAILURE at the discrete partition, and otherwise split the largest block
   with BFS-bisect;
4. (line 14) lift the abstract play to a play ``pi`` of ``M`` (Definition
   6.2.1);
5. (lines 15-16) if some agent misses its demand in the final state of
   ``pi``, refine with the sectors that the agent of smallest index with
   ``r_b(pi) < d(b)`` controls;
6. (lines 18-22) otherwise run VerifyNE: for each agent ``a``, in index
   order, one query ``smt(G_a(pi) and reward_a > r_a(pi))`` over the
   concrete encoding, in which the other agents follow the canonical
   completion of ``pi`` (Definition 4.7.1), without symmetry-breaking or
   candidate constraints.  If every query is unsatisfiable, return the
   canonical completion of ``pi`` (Definition 4.4.2).  A model ``beta`` is a
   counterexample, and ``D`` is the set of sectors whose control differs
   between the final states of ``pi`` and ``pi_beta`` (Definition 6.3.2);
7. (lines 24-34) return FAILURE if ``P`` is discrete; otherwise set ``P``
   to ``Refine(P, D)`` (Definition 6.3.3), or, if that adds no block,
   bisect the largest block of more than one sector inside ``D`` (the
   largest block of ``P`` if there is none).

Every iteration that does not return adds at least one block, so a run
returns after at most ``|T| - |P_0| + 1`` iterations (Theorem 6.4.2), and the
returned joint strategy is the canonical completion of a satisfying play
that passed VerifyNE, hence a demand-constrained Nash equilibrium of ``M``
(Theorem 6.4.1).  Neither property depends on the inner synthesiser, which
can be any :data:`SynthesiseFn`.

Symmetry reduction (Section 6.5) adds the lex-leader and spatial
canonicalisation constraints, computed from the automorphisms of the
weighted abstract graph and the demand classes, to the initial query of
each abstract synthesis call only; at the discrete partition the inner
synthesiser uses the constraints of Chapter 5 for ``M``, as a direct run
does.  No symmetry is taken from the coordinates of the synthetic abstract
territory, and VerifyNE is posed without symmetry constraints.

When Z3 answers a query with unknown, in the inner synthesiser or in
VerifyNE, the run stops with outcome ``"unk"`` (Section 7.1).  Outcomes:
``"ne"``, ``"fail"`` (FAILURE, ``reason`` ``"failure_no_candidate"`` from
line 9 or ``"failure_refuted"`` from line 25), ``"unk"``, ``"t/o"`` (the
optional wall-clock limit *time_limit_s*) and ``"n/c"`` (the optional cap
*max_refinements*; Algorithm 4 itself has neither limit).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from .abstraction import (
    AbstractRCMAS,
    Partition,
    abstract_symmetry_constraints,
    bfs_bisect,
    build_abstract_rcmas,
    deviation_set,
    lift_play,
    make_partition,
    refine_partition,
    sectors_of,
)
from .model import Coord, State, Territory, rewards_from_owner
from .smt_solve import (
    BaseEncoding,
    ExtraConstraints,
    SmtSolution,
    build_base_encoding,
    check_improvement_from_base,
)
from .strategy import CanonicalCompletion, Play, canonical_completion

OUTCOME_BY_REASON = {
    "ne": "ne",
    "failure_no_candidate": "fail",
    "failure_refuted": "fail",
    "unknown": "unk",
    "timeout": "t/o",
    "max_refinements": "n/c",
}

SYNTHESISERS = ("ibis", "qibis", "sibis")


# ---------------------------------------------------------------------------
# The inner synthesiser (line 6 of Algorithm 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SynthesisGame:
    """The game that ``SynthesiseNE`` receives in one iteration.

    Attributes:
        abstract: the abstract RCMAS of the current partition.
        territory: the synthetic abstract territory, or the territory of
            ``M`` at the discrete partition.
        num_agents: ``|Agt|``.
        horizon: ``m_hat`` (``m`` at the discrete partition).
        demands: ``d`` (``None`` = all 0).
        weights: block weights (``None`` at the discrete partition).
        custom_neighbors: block adjacency (``None`` at the discrete partition).
        symmetry: add the symmetry constraints of Chapter 5 for the grid
            territory to the initial query (discrete partition only).
        initial_constraints: constraints for the initial query only, here
            the symmetry constraints of the weighted abstract graph.
    """

    abstract: AbstractRCMAS
    territory: Territory
    num_agents: int
    horizon: int
    demands: tuple[int, ...] | None
    weights: tuple[int, ...] | None
    custom_neighbors: dict[int, list[int]] | None
    symmetry: bool
    initial_constraints: ExtraConstraints | None

    @property
    def is_concrete(self) -> bool:
        """True at the discrete partition, where the game is ``M`` itself."""
        return self.weights is None


@dataclass(frozen=True, slots=True)
class InnerResult:
    """What ``SynthesiseNE`` returns.

    Attributes:
        outcome: ``"ne"`` (``play`` is the play of the returned equilibrium
            of the game), ``"n/c"`` or ``"uns"`` (no candidate, ``bot`` in
            Algorithm 4) or ``"unk"`` (Z3 answered unknown).
        play: the play, over the block indices of the game.
        iterations: iterations of the inner synthesiser.
        rewards: rewards of ``play`` in the game (abstract rewards).
        detail: the synthesiser's own result object.
    """

    outcome: str
    play: Play | None = None
    iterations: int = 0
    rewards: tuple[int, ...] | None = None
    detail: Any = None


SynthesiseFn = Callable[[SynthesisGame], InnerResult]
"""``SynthesiseNE``: a game -> an equilibrium play of it, or no candidate."""


def make_synthesiser(
    name: str,
    *,
    max_iters: int = 30,
    timeout_ms: int | None = None,
    seed: int = 0,
    progress: bool = False,
) -> SynthesiseFn:
    """IBIS, S-IBIS or Q-IBIS as ``SynthesiseNE``, with the iteration bound ``N = max_iters``."""
    if name not in SYNTHESISERS:
        raise ValueError("synthesiser must be 'ibis', 'qibis', or 'sibis'")

    def run(game: SynthesisGame) -> InnerResult:
        res: Any
        if name == "qibis":
            from .qibis import QibisConfig, solve_qibis

            cfg = QibisConfig(
                max_iters=max_iters,
                timeout_ms=timeout_ms,
                symmetry=game.symmetry,
                demands=game.demands,
                seed=seed,
                progress=progress,
            )
            res = solve_qibis(
                territory=game.territory,
                num_agents=game.num_agents,
                horizon=game.horizon,
                cfg=cfg,
                weights=game.weights,
                custom_neighbors=game.custom_neighbors,
                initial_constraints=game.initial_constraints,
            )
        else:
            from .ibis import solve_ibis, solve_sibis

            solve = solve_sibis if name == "sibis" else solve_ibis
            res = solve(
                territory=game.territory,
                num_agents=game.num_agents,
                horizon=game.horizon,
                max_iters=max_iters,
                timeout_ms=timeout_ms,
                symmetry=game.symmetry,
                demands=game.demands,
                weights=game.weights,
                custom_neighbors=game.custom_neighbors,
                initial_constraints=game.initial_constraints,
                progress=progress,
            )
        play = res.play if res.outcome == "ne" else None
        return InnerResult(res.outcome, play, res.iterations, res.payoff_by_agent, res)

    return run


def synthesis_game(
    abstract: AbstractRCMAS,
    *,
    demands: tuple[int, ...] | None = None,
    symmetry: bool = False,
) -> SynthesisGame:
    """The game of line 6 for the abstract RCMAS *abstract*.

    At the discrete partition this is ``M`` with its own symmetry
    constraints (Chapter 5).  At every other partition it is the abstract
    game with block weights and block adjacency, and with symmetry on it
    carries the constraints of the weighted abstract graph for the initial
    query (Section 6.5).
    """
    n = abstract.num_agents
    if abstract.is_discrete:
        return SynthesisGame(
            abstract=abstract,
            territory=abstract.concrete_territory,
            num_agents=n,
            horizon=abstract.concrete_horizon,
            demands=demands,
            weights=None,
            custom_neighbors=None,
            symmetry=symmetry,
            initial_constraints=None,
        )
    init = abstract_symmetry_constraints(abstract, demands) if symmetry else None
    return SynthesisGame(
        abstract=abstract,
        territory=abstract.territory,
        num_agents=n,
        horizon=abstract.horizon,
        demands=demands,
        weights=abstract.weights,
        custom_neighbors=abstract.custom_neighbors,
        symmetry=False,
        initial_constraints=init,
    )


# ---------------------------------------------------------------------------
# VerifyNE (line 18 of Algorithm 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Result of VerifyNE.

    Attributes:
        status: ``"verified"``, ``"counterexample"`` or ``"unknown"``.
        agent: the deviating agent ``a`` (or the agent whose query Z3 could
            not decide).
        solution: the model ``beta``.
        deviation: the play ``pi_beta``.
        queries: number of queries posed (at most ``|Agt|``).
    """

    status: str
    agent: int | None = None
    solution: SmtSolution | None = None
    deviation: Play | None = None
    queries: int = 0


def verify_ne(
    base: BaseEncoding,
    play: Play,
    *,
    rewards: tuple[int, ...] | None = None,
    timeout_ms: int | None = None,
) -> VerifyResult:
    """VerifyNE of Section 6.3: one query per agent against the canonical completion.

    *base* is the formula ``F`` of ``M``, with the demands of ``M`` and
    without symmetry constraints; *play* must be satisfying (line 15), the
    hypothesis of Lemma 4.7.2.  For each agent ``a``, in index order, the
    query ``smt(G_a(pi) and reward_a > r_a(pi))`` is posed.  A model is a
    counterexample (Definition 6.3.1) and is returned at once; if every query
    is unsatisfiable the play is verified.  An unknown answer is returned as
    ``"unknown"`` and never read as unsatisfiable.
    """
    n = play.num_agents
    if base.num_agents != n or base.horizon != play.horizon:
        raise ValueError("the play does not belong to the encoded game")
    current = base.rewards_of(play.final_state) if rewards is None else rewards
    queries = 0
    for a in range(n):
        sol = check_improvement_from_base(
            base,
            agent=a,
            threshold=current[a],
            opponent_play=play,
            timeout_ms=timeout_ms,
        )
        queries += 1
        if sol.reason == "unknown":
            return VerifyResult("unknown", a, None, None, queries)
        if not sol.is_sat:
            continue
        assert sol.play is not None and sol.payoff_by_agent is not None
        if sol.payoff_by_agent[a] <= current[a]:
            raise RuntimeError(
                "a model of G_a(pi) with reward_a > r_a(pi) must improve "
                "on r_a(pi) (Lemma 4.7.2(1) violated)"
            )
        return VerifyResult("counterexample", a, sol, sol.play, queries)
    return VerifyResult("verified", None, None, None, queries)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CegarIteration:
    """One iteration of the loop of Algorithm 4.

    Attributes:
        index: the iteration (1-based).
        blocks: ``|P|`` at the start of the iteration.
        synthesis: outcome of the inner synthesiser (``"ne"``, ``"n/c"``,
            ``"uns"`` or ``"unk"``).
        inner_iterations: iterations of the inner synthesiser.
        abstract_rewards: rewards of the abstract play, if any.
        lifted_rewards: ``r_a(pi)`` of the lifted play, if any.
        check: ``"no_candidate"``, ``"unsatisfying"``, ``"refuted"``,
            ``"verified"`` or ``"unknown"``.
        agent: ``b`` (unsatisfying play) or the deviating agent ``a``.
        deviation_size: ``|D|`` for a refuted or unsatisfying play.
        refinement: ``"bisect"`` (lines 11-12), ``"refine"`` (line 33),
            ``"fallback"`` (lines 29-31) or ``None`` (the run stopped).
        blocks_after: ``|P|`` at the end of the iteration.
        verify_queries: number of VerifyNE queries.
        time_s: wall-clock time of the iteration.
    """

    index: int
    blocks: int
    synthesis: str
    inner_iterations: int
    abstract_rewards: tuple[int, ...] | None
    lifted_rewards: tuple[int, ...] | None
    check: str
    agent: int | None
    deviation_size: int | None
    refinement: str | None
    blocks_after: int
    verify_queries: int
    time_s: float


@dataclass(frozen=True, slots=True)
class CegarResult:
    """Result of CEGAR-NE.

    Attributes:
        is_sat: an equilibrium was returned (same as ``found_ne``).
        reason: ``"ne"``, ``"failure_no_candidate"``, ``"failure_refuted"``,
            ``"unknown"``, ``"timeout"`` or ``"max_refinements"``.
        found_ne: the canonical completion of ``play`` is a demand-constrained
            NE of ``M`` (Theorem 6.4.1).
        iterations: number of iterations of the loop that ran (at most
            ``|T| - |P_0| + 1``, Theorem 6.4.2).
        final_partition_size: ``|P|`` in the last iteration, the number of
            blocks of the final partition.
        strategy: the profiles of ``play`` as coordinates.
        payoff_by_agent: ``r_a(play)`` on the final state (Definition 3.9.2).
        final_solution: ``play`` as an :class:`SmtSolution` over the concrete
            territory (for rendering).
        play: the returned play if ``found_ne``; otherwise the last lifted
            play, if any.
        outcome: ``"ne"``, ``"fail"``, ``"unk"``, ``"t/o"`` or ``"n/c"``.
        at_discrete: the returned equilibrium was verified at the discrete
            partition.
        refinements: number of refinements of the partition.
        initial_partition_size: ``|P_0|``.
        inner_iterations: iterations of the inner synthesiser, summed over
            the iterations of the loop.
        partition: the final partition.
        trace: one :class:`CegarIteration` per iteration.

    The returned joint strategy is ``completion()``, the canonical completion
    of ``play`` (line 20 of Algorithm 4).
    """

    is_sat: bool
    reason: str
    found_ne: bool
    iterations: int
    final_partition_size: int
    strategy: tuple[tuple[Coord | None, ...], ...] | None
    payoff_by_agent: tuple[int, ...] | None
    final_solution: SmtSolution | None
    play: Play | None = None
    outcome: str = ""
    at_discrete: bool = False
    refinements: int = 0
    initial_partition_size: int = 0
    inner_iterations: int = 0
    partition: Partition | None = None
    trace: tuple[CegarIteration, ...] = ()

    @property
    def rewards(self) -> tuple[int, ...] | None:
        """``r_a(play)`` (alias of ``payoff_by_agent``)."""
        return self.payoff_by_agent

    @property
    def blocks(self) -> int:
        """Number of blocks of the final partition."""
        return self.final_partition_size

    def completion(self) -> CanonicalCompletion | None:
        """The canonical completion of the returned play, or ``None`` without an equilibrium."""
        if not self.found_ne or self.play is None:
            return None
        return canonical_completion(self.play)


def _solution_of(
    play: Play, territory: Territory, rewards: tuple[int, ...]
) -> SmtSolution:
    """A concrete play as an :class:`SmtSolution` (for rendering and legacy callers)."""
    final = State(
        territory=territory,
        num_agents=play.num_agents,
        owner_by_index=play.final_state,
        round_index=play.horizon,
    )
    return SmtSolution(
        is_sat=True,
        reason="sat",
        final_state=final,
        actions_by_round=play.to_coords(territory),
        owner_by_round=play.states,
        payoff_by_agent=rewards,
        play=play,
    )


# ---------------------------------------------------------------------------
# Algorithm 4
# ---------------------------------------------------------------------------


def solve_cegar(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int | None = None,
    initial_partition: str | Partition = "orbit",
    synthesiser: str | SynthesiseFn = "sibis",
    max_iters: int = 30,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
    symmetry: bool = False,
    demands: tuple[int, ...] | None = None,
    seed: int = 0,
    max_refinements: int | None = None,
    time_limit_s: float | None = None,
) -> CegarResult:
    """Run CEGAR-NE (Algorithm 4) on ``M = (Agt, T, d)``.

    Parameters
    ----------
    territory, num_agents, demands
        The RCMAS ``M``; *demands* ``None`` means every demand is 0.
    horizon
        Must be ``m = floor(|T|/|Agt|)``; ``None`` selects it.
    initial_partition
        ``P_0``: a :class:`Partition` or a name accepted by
        :func:`~rcmas.abstraction.make_partition` (default ``"orbit"``).
    synthesiser
        ``"sibis"`` (default, as in Chapter 7), ``"ibis"``, ``"qibis"``, or
        any :data:`SynthesiseFn`.
    max_iters
        The iteration bound ``N`` of the inner synthesiser, the same bound
        as a direct run (Section 7.2).  It does not bound the loop of
        Algorithm 4, which needs no bound (Theorem 6.4.2).
    timeout_ms
        Z3 timeout per query; an expired query answers unknown and the run
        stops with outcome ``"unk"``.
    symmetry
        Add symmetry-breaking constraints to the initial query of each
        inner synthesis call (Section 6.5).
    seed
        Seed of the Q-IBIS learners.
    max_refinements, time_limit_s
        Optional limits outside Algorithm 4: stop with outcome ``"n/c"``
        instead of refining more than *max_refinements* times, or with
        ``"t/o"`` at the start of an iteration once *time_limit_s* seconds
        have passed.
    progress, timing
        Log every iteration to stderr, with times.
    """
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if progress:
            if timing:
                msg = f"{msg} t={time.perf_counter() - t0:.3f}s"
            print(msg, file=sys.stderr, flush=True)

    n = num_agents
    if n <= 0:
        raise ValueError("num_agents must be >= 1")
    S = len(territory)
    m = S // n
    if horizon is None:
        horizon = m
    if horizon != m:
        raise ValueError(
            f"horizon must be m = floor(|T|/|Agt|) = {m} (Definition 3.9.1), got {horizon}"
        )
    if demands is not None:
        demands = tuple(demands)
        if len(demands) != n:
            raise ValueError("demands must have one entry per agent")
    if max_iters < 0:
        raise ValueError("max_iters must be >= 0")
    if isinstance(synthesiser, str):
        synth = make_synthesiser(
            synthesiser, max_iters=max_iters, timeout_ms=timeout_ms, seed=seed
        )
    else:
        synth = synthesiser

    partition = make_partition(initial_partition, territory, n)
    p0 = len(partition)
    bound = S - p0 + 1
    d = demands if demands is not None else (0,) * n
    neighbors = territory.neighbor_map()

    # F of M, built once and shared by every VerifyNE query (Section 7.1).
    base = build_base_encoding(
        territory=territory, num_agents=n, horizon=m, demands=demands
    )

    trace: list[CegarIteration] = []
    refinements = 0
    inner_total = 0
    last_play: Play | None = None
    last_rewards: tuple[int, ...] | None = None

    def _result(reason: str, it: int, k: int, ne: bool = False) -> CegarResult:
        play = last_play
        return CegarResult(
            is_sat=ne,
            reason=reason,
            found_ne=ne,
            iterations=it,
            final_partition_size=k,
            strategy=None if play is None else play.to_coords(territory),
            payoff_by_agent=last_rewards,
            final_solution=(
                None
                if play is None or last_rewards is None
                else _solution_of(play, territory, last_rewards)
            ),
            play=play,
            outcome=OUTCOME_BY_REASON[reason],
            at_discrete=ne and k == S,
            refinements=refinements,
            initial_partition_size=p0,
            inner_iterations=inner_total,
            partition=partition,
            trace=tuple(trace),
        )

    def _push(row: dict[str, Any], started: float, **fields: Any) -> None:
        row.update(fields)
        trace.append(CegarIteration(**row, time_s=time.perf_counter() - started))

    it = 0
    while True:
        it += 1
        if it > bound:  # pragma: no cover - Theorem 6.4.2
            raise RuntimeError("CEGAR-NE exceeded |T| - |P_0| + 1 iterations")
        if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
            _log(f"cegar iter={it} time limit reached")
            return _result("timeout", it - 1, len(partition))
        it_t0 = time.perf_counter()
        k = len(partition)
        row: dict[str, Any] = {
            "index": it,
            "blocks": k,
            "abstract_rewards": None,
            "lifted_rewards": None,
            "agent": None,
            "deviation_size": None,
            "refinement": None,
            "blocks_after": k,
            "verify_queries": 0,
        }

        # Line 5: the abstract RCMAS.  Line 6: SynthesiseNE.
        abstract = build_abstract_rcmas(territory, n, partition)
        game = synthesis_game(abstract, demands=demands, symmetry=symmetry)
        inner = synth(game)
        inner_total += inner.iterations
        row["synthesis"] = inner.outcome
        row["inner_iterations"] = inner.iterations
        _log(
            f"cegar iter={it} blocks={k} horizon={game.horizon} "
            f"synthesis={inner.outcome} inner_iters={inner.iterations} "
            f"abstract_rewards={inner.rewards}"
        )
        if inner.outcome == "unk":
            _push(row, it_t0, check="unknown")
            return _result("unknown", it, k)

        if inner.outcome != "ne" or inner.play is None:
            # Lines 7-12: no abstract candidate.
            if k == S:
                _push(row, it_t0, check="no_candidate")
                _log(f"cegar iter={it} no candidate at the discrete partition")
                return _result("failure_no_candidate", it, k)
            if max_refinements is not None and refinements >= max_refinements:
                _push(row, it_t0, check="no_candidate")
                return _result("max_refinements", it, k)
            star = partition.largest_block()
            refined = refine_partition(
                partition, bfs_bisect(partition.blocks[star], territory)
            )
            assert len(refined) == k + 1
            partition = refined
            refinements += 1
            _push(
                row,
                it_t0,
                check="no_candidate",
                refinement="bisect",
                blocks_after=len(partition),
            )
            _log(f"cegar iter={it} no candidate: bisect block {star}")
            continue

        # Line 14: lift.
        pi = lift_play(abstract, inner.play)
        r = rewards_from_owner(pi.final_state, n, neighbors)
        last_play, last_rewards = pi, r
        row["abstract_rewards"] = inner.rewards
        row["lifted_rewards"] = r
        _log(f"cegar iter={it} lifted_rewards={r}")

        # Lines 15-16: the satisfaction check on the lifted play.
        unsatisfied = [b for b in range(n) if r[b] < d[b]]
        if unsatisfied:
            agent = unsatisfied[0]
            D = sectors_of(pi.final_state, agent)
            row["check"] = "unsatisfying"
            _log(f"cegar iter={it} lifted play not satisfying: agent={agent}")
        else:
            # Line 18: VerifyNE, without symmetry constraints.
            vr = verify_ne(base, pi, rewards=r, timeout_ms=timeout_ms)
            row["verify_queries"] = vr.queries
            if vr.status == "unknown":
                _push(row, it_t0, check="unknown", agent=vr.agent)
                _log(f"cegar iter={it} VerifyNE: solver answered unknown")
                return _result("unknown", it, k)
            if vr.status == "verified":
                _push(row, it_t0, check="verified")
                _log(f"cegar iter={it} verified with {k} blocks")
                return _result("ne", it, k, ne=True)
            assert vr.deviation is not None and vr.agent is not None
            agent = vr.agent
            D = deviation_set(pi.final_state, vr.deviation.final_state)
            assert D, "a profitable deviation changes the final state"
            row["check"] = "refuted"
            _log(f"cegar iter={it} counterexample: agent={agent} |D|={len(D)}")
        row["agent"] = agent
        row["deviation_size"] = len(D)

        # Line 24: refuted at the discrete partition.
        if k == S:
            _push(row, it_t0)
            _log(f"cegar iter={it} refuted at the discrete partition")
            return _result("failure_refuted", it, k)
        if max_refinements is not None and refinements >= max_refinements:
            _push(row, it_t0)
            return _result("max_refinements", it, k)

        # Lines 27-34: refine, with the fallback when D is a union of blocks.
        refined = refine_partition(partition, D)
        how = "refine"
        if len(refined) == k:
            inside = [
                p for p, B in enumerate(partition.blocks) if B <= D and len(B) > 1
            ]
            star = (
                partition.largest_block(inside) if inside else partition.largest_block()
            )
            refined = refine_partition(
                partition, bfs_bisect(partition.blocks[star], territory)
            )
            how = "fallback"
        assert len(refined) > k
        partition = refined
        refinements += 1
        _push(row, it_t0, refinement=how, blocks_after=len(partition))
        _log(f"cegar iter={it} {how}: {k} -> {len(partition)} blocks")
