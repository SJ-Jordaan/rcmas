"""Q-IBIS: Q-learning-guided iterative synthesis (thesis Algorithm 3, Section 4.9).

Q-IBIS runs the loop of IBIS (Algorithm 1) on a single play, but restricts
each best-response query to a few candidate actions that Q-learners rank
highest, and checks every candidate equilibrium with unrestricted queries
before it returns it:

1. (line 1) every agent's Q-table is pre-trained against uniformly random
   opponents for ``E_off`` episodes;
2. (lines 2-6) the initial play ``pi^0`` is the play of a model of
   ``smt(F)`` or, with ``initial="collective"``, of the collectively
   optimal query ``smt(F, f)`` (Definition 4.6.1); with symmetry reduction
   on, this is the only query that carries the symmetry-breaking
   constraints (Section 5.4.5, Section 7.1);
3. (lines 8-10) in iteration ``i`` the run stops if ``pi^i`` occurred
   before (outcome ``"n/c"``);
4. (line 11) every Q-table is trained further for ``E_on`` episodes while
   the other agents follow the canonical completion ``alpha^i`` of ``pi^i``
   (Definition 4.4.2);
5. (lines 13-15) for every agent ``a``, ``V^i_a`` is the set of live states
   visited in that training together with ``s^i_0, ..., s^i_{m-1}``, and
   ``Cand^{a,i}(s)`` is ``TopK(Q_a, s, k)`` (Definition 2.4.4) plus, at
   ``s = s^i_t``, the current action ``ap^i_t(a)``.  The restricted query
   ``smt(G_a(pi^i) and [Cand^{a,i}], reward_a)`` is posed with the
   state-guarded candidate constraint, and
   ``IP^i(a) = r_a(pi_{beta_a}) - r_a(pi^i)`` on final states.  The
   assignment that encodes ``pi^i`` is a model of the restricted query
   (Corollary 4.3.3(3)), so the query never returns nil and ``IP^i(a) >= 0``;
   a violation of either is a bug and raises :class:`RuntimeError`;
6. (lines 17-25) if every restricted potential is 0, a certification round
   poses the unrestricted best-response query ``smt(G_a(pi^i), reward_a)``
   (Definition 4.7.1) for every agent.  If every potential is still 0, the
   canonical completion of ``pi^i`` is a demand-constrained Nash equilibrium
   (Lemma 4.7.2(4), Corollary 4.9.1) and the run returns it; otherwise the
   update uses the unrestricted potentials;
7. (lines 26-27) the agent with the largest potential, the smallest index
   on ties, replaces the play by its best-response play.

In every query the opponents of the deviating agent follow the canonical
completion of the current play, through the opponent constraint
``[Fix^a_pi]`` (Remark 4.9.1).  No query except the initial one contains
symmetry-breaking constraints.  When Z3 answers a query with unknown, the
run stops with outcome ``"unk"`` and never reads the answer as the absence
of an improvement (Section 7.1).

The loop runs for ``i = 0, ..., N``, or without a bound if ``N`` is
``None``; an unbounded loop ends at an equilibrium or at the first
recurring play, since a game has finitely many plays.

The result has the fields of :class:`~rcmas.ibis.IbisResult`: ``play``,
``rewards`` (``payoff_by_agent``), ``outcome`` (``"ne"``, ``"n/c"``,
``"unk"`` or ``"uns"``), ``reason``, ``iterations``, the index ``i`` of
the iteration in which the run stopped (``N`` if the bound is reached), and
the initial play with the time of its query.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

from .ibis import (
    OUTCOME_BY_REASON,
    BestResponse,
    BestResponseFn,
    InitialHook,
    failed_initial,
    initial_play_of,
    iteration_indices,
    smt_best_response,
    with_initial,
)
from .model import Coord, Territory
from .qlearning import (
    LearningGame,
    LearningParams,
    QValues,
    StateKey,
    top_k,
    train_q,
)
from .smt_solve import (
    INITIAL_QUERIES,
    BaseEncoding,
    ExtraConstraints,
    SmtSolution,
    best_response_from_base,
    build_base_encoding,
)
from .strategy import ActionCandidates, CanonicalCompletion, Play, canonical_completion

SectorCandidates = dict[StateKey, tuple[int, ...]]
"""``Cand^{a,i}``: state -> indices of the candidate sectors of one agent."""

RefineFn = Callable[[Play], list[SectorCandidates]]
"""Line 11 and line 13 of Algorithm 3: train against the completion of a play, return ``Cand^{a,i}`` per agent."""

RestrictedBestResponseFn = Callable[
    [Play, int, tuple[int, ...], SectorCandidates], BestResponse
]
"""``(pi, a, rewards of pi, Cand^{a,i}) -> beta_a`` of the restricted query (line 14)."""


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QibisConfig:
    """Parameters of Q-IBIS; the defaults are those of Section 4.9.

    ``max_iters`` and ``initial`` default to the first run of Chapter 7
    (``N = 30``, ``smt(F)``); the revised protocol of
    ``experiments/thesis-eval.toml`` sets ``max_iters=None`` and
    ``initial="collective"``.

    Attributes:
        max_iters: the iteration bound ``N`` (the loop runs ``i = 0..N``);
            ``None`` runs the loop without a bound.
        initial: the query for the initial play, ``"satisfying"``
            (``smt(F)``) or ``"collective"`` (``smt(F, f)``).
        top_k: ``k`` of ``TopK``.
        offline_episodes: ``E_off``, episodes per agent of the pre-training.
        online_episodes: ``E_on``, episodes per agent in every iteration.
        learning_rate: ``eta``.
        discount: ``gamma``.
        epsilon_start, epsilon_end: linear exploration schedule of every
            training run.
        step_reward, growth_reward, terminal_reward, collision_penalty,
            normalize_rewards: the shaped reward of Section 4.9.
        timeout_ms: Z3 timeout per query (an expired query answers unknown).
        progress, timing: log every iteration to stderr, with times.
        seed: seed of the learners' random number generators.
        symmetry: add the symmetry-breaking constraints to the initial query.
        demands: ``d(a)`` for every agent (``None`` = all 0).
    """

    max_iters: int | None = 30
    initial: str = "satisfying"
    top_k: int = 3
    offline_episodes: int = 800
    online_episodes: int = 1000
    learning_rate: float = 0.1
    discount: float = 0.99
    epsilon_start: float = 0.7
    epsilon_end: float = 0.05
    step_reward: float = 0.1
    growth_reward: float = 1.0
    terminal_reward: float = 5.0
    collision_penalty: float = 10.0
    normalize_rewards: bool = True
    timeout_ms: int | None = None
    progress: bool = False
    timing: bool = False
    seed: int = 0
    symmetry: bool = False
    demands: tuple[int, ...] | None = None

    def learning_params(self) -> LearningParams:
        """The hyperparameters of the learner."""
        return LearningParams(
            learning_rate=self.learning_rate,
            discount=self.discount,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            step_reward=self.step_reward,
            growth_reward=self.growth_reward,
            terminal_reward=self.terminal_reward,
            collision_penalty=self.collision_penalty,
            normalize=self.normalize_rewards,
        )


@dataclass(frozen=True, slots=True)
class QibisIteration:
    """One iteration of Algorithm 3.

    Attributes:
        index: the iteration ``i``.
        rewards: ``r_a(pi^i)`` for every agent.
        restricted: ``IP^i(a)`` of the restricted queries (line 15); shorter
            than ``|Agt|`` if the iteration stopped early.
        certified: ``IP^i(a)`` of the certification round (line 20), or
            ``()`` if the round was not posed.
        chosen: ``a*``, or ``None`` if the play was not updated.
        checks: number of solver checks posed in the iteration.
        candidate_states: ``|V^i_a|`` for every agent.
    """

    index: int
    rewards: tuple[int, ...]
    restricted: tuple[int, ...]
    certified: tuple[int, ...]
    chosen: int | None
    checks: int
    candidate_states: tuple[int, ...] = ()

    @property
    def improvement(self) -> tuple[int, ...]:
        """The potentials that decide the iteration: certified if posed, else restricted."""
        return self.certified if self.certified else self.restricted


@dataclass(frozen=True, slots=True)
class QibisResult:
    """Result of Q-IBIS, with the fields of :class:`~rcmas.ibis.IbisResult`.

    Attributes:
        is_sat: a satisfying play was found (``play`` is set).
        reason: ``"ne"``, ``"cycle"``, ``"max_iters"``, ``"unknown"`` or ``"unsat"``.
        found_ne: the canonical completion of ``play`` is an equilibrium.
        iterations: index ``i`` of the last iteration that ran (``N`` if the
            bound was reached, with ``play`` the play produced in iteration ``N``).
        strategy: the profiles of ``play`` as coordinates.
        payoff_by_agent: ``r_a(play)``, evaluated on the final state.
        final_solution: the SMT model whose play is ``play`` (``None`` for a
            given initial play).
        play: the final play ``pi^i`` (tool sector indices).
        outcome: ``"ne"``, ``"n/c"``, ``"unk"`` or ``"uns"``.
        trace: one :class:`QibisIteration` per iteration.
        initial, initial_query_s, initial_play, initial_rewards: the initial
            query and its play, as in :class:`~rcmas.ibis.IbisResult`.

    The returned joint strategy is ``completion()``, the canonical
    completion of ``play`` (Definition 4.4.2).
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
    trace: tuple[QibisIteration, ...] = ()
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


# ---------------------------------------------------------------------------
# The learners (lines 1, 11 and 13)
# ---------------------------------------------------------------------------


class QibisLearner:
    """The Q-tables ``Q_a`` of a run and the candidate sets they induce.

    Each agent has its own table and its own random number generator, seeded
    from *seed*, so a run is reproducible.  The tables persist over the
    whole run: :meth:`pretrain` is line 1 of Algorithm 3, and
    :meth:`refine` is lines 11 and 13 of every iteration.
    """

    def __init__(
        self,
        game: LearningGame,
        params: LearningParams,
        *,
        k: int,
        seed: int = 0,
    ) -> None:
        if k < 0:
            raise ValueError("top_k must be >= 0")
        self.game = game
        self.params = params
        self.k = k
        self.tables: list[QValues] = [{} for _ in range(game.num_agents)]
        self.rngs = [random.Random(f"qibis:{seed}:{a}") for a in range(game.num_agents)]

    def pretrain(self, episodes: int) -> None:
        """Line 1: train every ``Q_a`` against uniformly random opponents."""
        for a in range(self.game.num_agents):
            train_q(
                self.tables[a],
                self.game,
                a,
                opponents=None,
                episodes=episodes,
                params=self.params,
                rng=self.rngs[a],
            )

    def candidates(
        self, play: Play, agent: int, visited: set[StateKey]
    ) -> SectorCandidates:
        """``Cand^{a,i}`` over ``V^i_a = visited + {s_0, ..., s_{m-1}}`` (line 13)."""
        q = self.tables[agent]
        out: SectorCandidates = {}
        states = set(visited)
        states.update(play.states[: play.horizon])
        for s in states:
            chosen = set(top_k(q, s, self.game.options(s), self.k))
            t = play.step_of(s)
            if t is not None:
                chosen.add(play.actions[t][agent])
            out[s] = tuple(sorted(chosen))
        return out

    def refine(self, play: Play, episodes: int) -> list[SectorCandidates]:
        """Line 11 and line 13: train every ``Q_a`` against the completion of *play*.

        The other agents follow the canonical completion of *play*.  Returns
        ``Cand^{a,i}`` for every agent.
        """
        completion = canonical_completion(play)
        out: list[SectorCandidates] = []
        for a in range(self.game.num_agents):
            visited = train_q(
                self.tables[a],
                self.game,
                a,
                opponents=completion.action,
                episodes=episodes,
                params=self.params,
                rng=self.rngs[a],
            )
            out.append(self.candidates(play, a, visited))
        return out


# ---------------------------------------------------------------------------
# The restricted query (line 14)
# ---------------------------------------------------------------------------


def restricted_smt_best_response(
    base: BaseEncoding, timeout_ms: int | None = None
) -> RestrictedBestResponseFn:
    """The restricted query ``smt(G_a(pi) and [Cand^{a,i}], reward_a)`` (line 14).

    The candidate constraint is state-guarded: at a candidate state ``s``
    reached at step ``t`` the action of ``a`` is one of ``Cand^{a,i}(s)``,
    and ``a`` is free at every other state.
    """
    sectors = base.territory.ordered_sectors()
    n = base.num_agents

    def br(
        play: Play, agent: int, current: tuple[int, ...], cand: SectorCandidates
    ) -> BestResponse:
        coords: ActionCandidates = {
            s: tuple(sectors[j] for j in js) for s, js in cand.items()
        }
        per_agent = tuple(coords if b == agent else None for b in range(n))
        sol = best_response_from_base(
            base,
            play,
            agent,
            action_candidates_by_agent=per_agent,
            timeout_ms=timeout_ms,
        )
        if sol.reason == "unknown":
            return BestResponse("unknown", current[agent], None, None, 1)
        if not sol.is_sat:
            raise RuntimeError(
                "restricted best-response query unsatisfiable, although the current "
                "play is a model of it (Section 4.9)"
            )
        assert sol.play is not None and sol.payoff_by_agent is not None
        return BestResponse("ok", sol.payoff_by_agent[agent], sol.play, sol, 1)

    return br


# ---------------------------------------------------------------------------
# The loop of Algorithm 3 (lines 7-29)
# ---------------------------------------------------------------------------


def run_qibis_loop(
    *,
    initial_play: Play,
    initial_solution: SmtSolution | None,
    refine: RefineFn,
    restricted_best_response: RestrictedBestResponseFn,
    best_response: BestResponseFn,
    rewards_of: Callable[[StateKey], tuple[int, ...]],
    max_iters: int | None,
    territory: Territory | None = None,
    log: Callable[[str], None] | None = None,
) -> QibisResult:
    """Run lines 7-29 of Algorithm 3 from *initial_play*.

    *refine* trains the learners against the completion of a play and
    returns the candidate sets (lines 11 and 13); *restricted_best_response*
    answers the restricted query (line 14); *best_response* answers the
    unrestricted query of the certification round (line 19).  *rewards_of*
    evaluates ``r_a`` on a final control vector.  The loop runs for
    ``i = 0, ..., max_iters``, or until it returns if *max_iters* is
    ``None``.
    """

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    def _result(
        reason: str, play: Play, sol: SmtSolution | None, iterations: int
    ) -> QibisResult:
        coords = play.to_coords(territory) if territory is not None else None
        return QibisResult(
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

    def _potential(
        br: BestResponse, a: int, current: tuple[int, ...], kind: str
    ) -> int:
        ip = br.reward - current[a]
        if ip < 0:
            raise RuntimeError(
                f"negative improvement potential for agent {a} in the {kind} query "
                "(Section 4.9 and Lemma 4.7.2(3) violated)"
            )
        return ip

    play = initial_play
    solution = initial_solution
    seen: set[Play] = set()
    trace: list[QibisIteration] = []
    n = play.num_agents

    for i in iteration_indices(max_iters):
        current = rewards_of(play.final_state)
        if play in seen:
            _log(f"iter={i} play recurs")
            trace.append(QibisIteration(i, current, (), (), None, 0))
            return _result("cycle", play, solution, i)
        seen.add(play)

        candidates = refine(play)
        sizes = tuple(len(c) for c in candidates)
        _log(f"iter={i} rewards={current} candidate_states={sizes}")

        restricted: list[int] = []
        responses: list[BestResponse] = []
        checks = 0
        for a in range(n):
            br = restricted_best_response(play, a, current, candidates[a])
            checks += br.checks
            if br.status == "unknown":
                _log(f"iter={i} agent={a} restricted query: solver answered unknown")
                trace.append(
                    QibisIteration(
                        i, current, tuple(restricted), (), None, checks, sizes
                    )
                )
                return _result("unknown", play, solution, i)
            restricted.append(_potential(br, a, current, "restricted"))
            responses.append(br)
            _log(f"iter={i} agent={a} restricted best_response={br.reward}")

        certified: list[int] = []
        if all(ip == 0 for ip in restricted):
            responses = []
            for a in range(n):
                br = best_response(play, a, current)
                checks += br.checks
                if br.status == "unknown":
                    _log(f"iter={i} agent={a} certification: solver answered unknown")
                    trace.append(
                        QibisIteration(
                            i,
                            current,
                            tuple(restricted),
                            tuple(certified),
                            None,
                            checks,
                            sizes,
                        )
                    )
                    return _result("unknown", play, solution, i)
                certified.append(_potential(br, a, current, "certification"))
                responses.append(br)
                _log(f"iter={i} agent={a} certified best_response={br.reward}")
            if all(ip == 0 for ip in certified):
                trace.append(
                    QibisIteration(
                        i,
                        current,
                        tuple(restricted),
                        tuple(certified),
                        None,
                        checks,
                        sizes,
                    )
                )
                _log(f"iter={i} rewards={current} equilibrium certified")
                return _result("ne", play, solution, i)
            potentials = certified
        else:
            potentials = restricted

        chosen = max(range(n), key=lambda a: (potentials[a], -a))
        trace.append(
            QibisIteration(
                i, current, tuple(restricted), tuple(certified), chosen, checks, sizes
            )
        )
        new_play = responses[chosen].play
        assert new_play is not None
        _log(f"iter={i} ip={tuple(potentials)} update agent={chosen}")
        play = new_play
        solution = responses[chosen].solution

    assert max_iters is not None  # an unbounded loop returns inside
    return _result("max_iters", play, solution, max_iters)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _validate(
    territory: Territory, num_agents: int, horizon: int | None, cfg: QibisConfig
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
    if cfg.max_iters is not None and cfg.max_iters < 0:
        raise ValueError("max_iters must be >= 0 or None")
    if cfg.initial not in INITIAL_QUERIES:
        raise ValueError(
            f"initial must be one of {INITIAL_QUERIES}, got {cfg.initial!r}"
        )
    if cfg.top_k < 0:
        raise ValueError("top_k must be >= 0")
    if cfg.offline_episodes < 0 or cfg.online_episodes < 0:
        raise ValueError("the numbers of episodes must be >= 0")
    return horizon


def solve_qibis(
    *,
    territory: Territory,
    num_agents: int,
    horizon: int | None = None,
    cfg: QibisConfig = QibisConfig(),
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    weight_balance_target: int | None = None,
    initial_constraints: ExtraConstraints | None = None,
    initial_play: Play | None = None,
    on_initial: InitialHook | None = None,
) -> QibisResult:
    """Run Q-IBIS (Algorithm 3) with the parameters of *cfg*.

    *horizon* must be ``m = floor(|T|/|Agt|)`` (``None`` selects it).
    *weights* and *custom_neighbors* describe an abstract RCMAS (Definition
    6.1.2); grid symmetry is then skipped, and the symmetry constraints of
    the abstract game can be passed as *initial_constraints*, which are
    added to the initial query only.  ``cfg.initial`` selects the initial
    query, ``smt(F)`` or ``smt(F, f)``; *initial_play* replaces it.
    *on_initial* is called with the outcome of the initial query as soon as
    it is known.
    """
    h = _validate(territory, num_agents, horizon, cfg)
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if cfg.progress:
            if cfg.timing:
                msg = f"{msg} t={time.perf_counter() - t0:.3f}s"
            print(msg, file=sys.stderr, flush=True)

    base = build_base_encoding(
        territory=territory,
        num_agents=num_agents,
        horizon=h,
        demands=cfg.demands,
        weights=weights,
        custom_neighbors=custom_neighbors,
        weight_balance_target=weight_balance_target,
    )
    game = LearningGame.from_territory(
        territory, num_agents, weights=weights, custom_neighbors=custom_neighbors
    )
    learner = QibisLearner(game, cfg.learning_params(), k=cfg.top_k, seed=cfg.seed)

    # Line 1: offline pre-training against random opponents.
    learner.pretrain(cfg.offline_episodes)
    _log(f"pretrained episodes={cfg.offline_episodes} per agent")

    # Lines 2-6: the initial play, from smt(F) or smt(F, f).
    abstract = weights is not None or custom_neighbors is not None
    if (
        initial_play is None
        and cfg.symmetry
        and abstract
        and initial_constraints is None
    ):
        _log(
            "symmetry: no grid symmetry for an abstract game; pass initial_constraints"
        )
    init = initial_play_of(
        base,
        initial=cfg.initial,
        symmetry=cfg.symmetry and not abstract,
        extra_constraints=initial_constraints,
        timeout_ms=cfg.timeout_ms,
        given=initial_play,
    )
    if on_initial is not None:
        on_initial(init)
    if init.reason == "unknown":
        _log(f"initial query ({cfg.initial}): solver answered unknown")
        return failed_initial(QibisResult, init)
    if init.play is None:
        _log("initial query unsatisfiable: no satisfying strategy exists")
        return failed_initial(QibisResult, init)
    if init.query_s is not None:
        _log(
            f"initial query ({cfg.initial}): rewards={init.rewards} "
            f"in {init.query_s:.3f}s"
        )

    res = run_qibis_loop(
        initial_play=init.play,
        initial_solution=init.solution,
        refine=lambda play: learner.refine(play, cfg.online_episodes),
        restricted_best_response=restricted_smt_best_response(base, cfg.timeout_ms),
        best_response=smt_best_response(base, cfg.timeout_ms),
        rewards_of=base.rewards_of,
        max_iters=cfg.max_iters,
        territory=territory,
        log=_log if cfg.progress else None,
    )
    return with_initial(res, init)
