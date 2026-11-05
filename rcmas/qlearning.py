"""Tabular Q-learning for RCMAS (thesis Sections 2.4 and 4.9).

The first part of the module is the learner of Q-IBIS (Section 4.9).  It
works on sector indices, so the same code serves a grid territory and the
abstract RCMAS of Definition 6.1.2 (block adjacency and block weights):

- each agent ``a`` has its own Q-table ``Q_a``, as in independent
  Q-learning (Section 2.4.3), indexed by the control vector
  ``(s(1), ..., s(|T|))`` of a state and the index of a claimed sector
  (:data:`QValues`);
- an episode (:func:`run_episode`) is a run from ``s_0`` that ends in a
  terminal state, a state that is not live (Definition 3.9.1), or after a
  collision.  While ``Q_a`` is trained the other agents either choose
  uniformly random available actions (offline pre-training) or follow a
  given joint strategy, in Q-IBIS the canonical completion of the current
  play (online refinement);
- ``a`` follows the epsilon-greedy policy of Definition 2.4.3, with an
  epsilon that decreases linearly from ``epsilon_start`` to ``epsilon_end``
  over each training run (:func:`train_q`); greedy ties go to the smallest
  sector index;
- the shaped reward of a round from ``s`` to ``s'`` without collision is
  ``0.1 + (r_a(s') - r_a(s))/|T|``, plus ``5 r_a(s')/|T|`` if ``s'`` is
  terminal; a collision gives ``-10`` to each agent involved and ``0`` to
  the others (:func:`shaped_reward`, :func:`collision_reward`);
- the update is Equation (2.1) with a constant learning rate ``eta`` and
  discount ``gamma`` (:func:`q_update`); the value of a terminal state, and
  of the failure state after a collision, is 0;
- :func:`top_k` gives ``TopK(Q, s, k)`` of Definition 2.4.4, with ties
  broken by the sector index.

The shaping is not potential-based (Section 2.4.4), and the constant
learning rate does not satisfy the conditions of Watkins and Dayan, so the
Q-values are used only to rank candidate actions; the soundness of Q-IBIS
does not depend on them (Corollary 4.9.1).

The second part keeps the coordinate-based helpers and the standalone
self-play trainer behind ``rcmas train``, which is not part of Q-IBIS.
"""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .model import (
    UNOWNED,
    Coord,
    State,
    Territory,
    evolve,
    largest_component_weight,
    largest_region_size,
    scores,
)

StateKey = tuple[int, ...]
"""A control vector (``UNOWNED`` = -1 for an unoccupied sector)."""

QValues = dict[StateKey, dict[int, float]]
"""A Q-table: control vector -> sector index -> Q-value (missing entries are 0)."""

OpponentFn = Callable[[StateKey, int], int]
"""``(state, agent) -> index of the sector the agent claims`` at a live state."""


# ===========================================================================
# Part 1: the learner of Q-IBIS (Section 4.9)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class LearningGame:
    """The RCMAS in which a Q-learner of Q-IBIS is trained.

    Attributes:
        num_sectors: ``|T|`` (the number of blocks in an abstract game).
        num_agents: ``|Agt|``.
        neighbors: sector adjacency (four-connected, or block adjacency).
        weights: sector weights (``None`` = unit weights).
    """

    num_sectors: int
    num_agents: int
    neighbors: dict[int, frozenset[int]] | dict[int, list[int]]
    weights: tuple[int, ...] | None = None

    @staticmethod
    def from_territory(
        territory: Territory,
        num_agents: int,
        *,
        weights: tuple[int, ...] | None = None,
        custom_neighbors: dict[int, list[int]] | None = None,
    ) -> LearningGame:
        """The game on *territory*, with *custom_neighbors* and *weights* for an abstract RCMAS."""
        if num_agents <= 0:
            raise ValueError("num_agents must be >= 1")
        if weights is not None and len(weights) != len(territory):
            raise ValueError("weights must have one entry per sector")
        neighbors = (
            territory.neighbor_map() if custom_neighbors is None else custom_neighbors
        )
        return LearningGame(len(territory), num_agents, neighbors, weights)

    @property
    def initial_state(self) -> StateKey:
        """``s_0``: every sector unoccupied."""
        return (UNOWNED,) * self.num_sectors

    def options(self, state: StateKey) -> tuple[int, ...]:
        """``A_s``: the unoccupied sectors of *state*, in the enumeration order."""
        return tuple(i for i, o in enumerate(state) if o == UNOWNED)

    def is_live(self, state: StateKey) -> bool:
        """At least ``|Agt|`` unoccupied sectors (Definition 3.8.1)."""
        return sum(1 for o in state if o == UNOWNED) >= self.num_agents

    def reward(self, state: StateKey, agent: int) -> int:
        """``r_a(s)``: the weight of the largest cohesive region of *agent* (Definition 3.9.2)."""
        return largest_component_weight(state, agent, self.neighbors, self.weights)


@dataclass(frozen=True, slots=True)
class LearningParams:
    """Hyperparameters of the learner (Section 4.9).

    Attributes:
        learning_rate: ``eta`` of Equation (2.1), constant over a run.
        discount: ``gamma``.
        epsilon_start, epsilon_end: the linear exploration schedule.
        step_reward: reward of a round without collision (``0.1``).
        growth_reward: coefficient of ``(r_a(s') - r_a(s))/|T|``.
        terminal_reward: coefficient of ``r_a(s')/|T|`` at a terminal ``s'``.
        collision_penalty: ``10``; a collision gives ``-10`` to each agent
            involved.
        normalize: divide the reward terms by ``|T|`` (the thesis setting).
    """

    learning_rate: float = 0.1
    discount: float = 0.99
    epsilon_start: float = 0.7
    epsilon_end: float = 0.05
    step_reward: float = 0.1
    growth_reward: float = 1.0
    terminal_reward: float = 5.0
    collision_penalty: float = 10.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        for eps in (self.epsilon_start, self.epsilon_end):
            if not 0.0 <= eps <= 1.0:
                raise ValueError("epsilon must lie in [0, 1]")


def collides(profile: tuple[int, ...]) -> bool:
    """True if two agents claim the same sector (Definition 3.6.1)."""
    return len(set(profile)) < len(profile)


def collision_reward(
    profile: tuple[int, ...], agent: int, params: LearningParams
) -> float:
    """Shaped reward of *agent* for a round that collides: ``-10`` if involved, else ``0``."""
    if profile.count(profile[agent]) > 1:
        return -params.collision_penalty
    return 0.0


def shaped_reward(
    game: LearningGame,
    prev: StateKey,
    nxt: StateKey,
    agent: int,
    params: LearningParams,
    *,
    prev_reward: int | None = None,
    next_reward: int | None = None,
) -> float:
    """Shaped reward of *agent* for a round from *prev* to *nxt* without collision.

    ``0.1 + (r_a(s') - r_a(s))/|T|``, plus ``5 r_a(s')/|T|`` if ``s'`` is
    terminal (not live).  *prev_reward* and *next_reward* may pass
    ``r_a(s)`` and ``r_a(s')`` if they are already known.
    """
    norm = float(game.num_sectors) if params.normalize and game.num_sectors > 0 else 1.0
    r_prev = game.reward(prev, agent) if prev_reward is None else prev_reward
    r_next = game.reward(nxt, agent) if next_reward is None else next_reward
    r = params.step_reward + params.growth_reward * float(r_next - r_prev) / norm
    if not game.is_live(nxt):
        r += params.terminal_reward * float(r_next) / norm
    return r


def greedy_action(q: QValues, state: StateKey, options: tuple[int, ...]) -> int:
    """``argmax_j Q(s, j)`` over *options*, the smallest sector index on ties."""
    if not options:
        raise ValueError("no action is available")
    row = q.get(state)
    if not row:
        return options[0]
    best = options[0]
    best_v = row.get(best, 0.0)
    for j in options[1:]:
        v = row.get(j, 0.0)
        if v > best_v:
            best, best_v = j, v
    return best


def epsilon_greedy_action(
    q: QValues,
    state: StateKey,
    options: tuple[int, ...],
    epsilon: float,
    rng: random.Random,
) -> int:
    """The epsilon-greedy policy of Definition 2.4.3."""
    if not options:
        raise ValueError("no action is available")
    if rng.random() < epsilon:
        return rng.choice(options)
    return greedy_action(q, state, options)


def top_k(
    q: QValues, state: StateKey, options: tuple[int, ...], k: int
) -> tuple[int, ...]:
    """``TopK(Q, s, k)`` of Definition 2.4.4.

    The first ``min(k, |A_s|)`` actions of *options* ordered by decreasing
    ``Q(s, .)``, with ties broken by the sector index.
    """
    if k <= 0:
        return ()
    row = q.get(state, {})
    ranked = sorted(options, key=lambda j: (-row.get(j, 0.0), j))
    return tuple(ranked[:k])


def max_q(q: QValues, state: StateKey, options: tuple[int, ...]) -> float:
    """``max_{j in A_s} Q(s, j)`` (0 for a state without entries)."""
    row = q.get(state)
    if row is None or not options:
        return 0.0
    return max(row.get(j, 0.0) for j in options)


def q_update(
    q: QValues,
    game: LearningGame,
    state: StateKey,
    action: int,
    reward: float,
    next_state: StateKey | None,
    params: LearningParams,
) -> float:
    """Equation (2.1): ``Q(s,j) <- (1-eta) Q(s,j) + eta (r + gamma max_j' Q(s',j'))``.

    *next_state* is ``None`` after a collision.  A terminal ``s'`` and the
    failure state have no actions, so their value is 0.  Returns the new
    value.
    """
    if next_state is None or not game.is_live(next_state):
        future = 0.0
    else:
        future = max_q(q, next_state, game.options(next_state))
    row = q.setdefault(state, {})
    old = row.get(action, 0.0)
    new = (1.0 - params.learning_rate) * old + params.learning_rate * (
        reward + params.discount * future
    )
    row[action] = new
    return new


@dataclass(frozen=True, slots=True)
class Transition:
    """One round of an episode, from the point of view of the learning agent.

    Attributes:
        state: the live state ``s`` in which the round starts.
        profile: the sectors claimed by all agents.
        next_state: ``s'``, or ``None`` if the profile collides.
        reward: the shaped reward of the learning agent.
    """

    state: StateKey
    profile: tuple[int, ...]
    next_state: StateKey | None
    reward: float


def run_episode(
    q: QValues,
    game: LearningGame,
    agent: int,
    *,
    opponents: OpponentFn | None,
    epsilon: float,
    params: LearningParams,
    rng: random.Random,
) -> tuple[Transition, ...]:
    """Play one episode from ``s_0`` and update ``Q_agent`` after every round.

    *opponents* gives the actions of the other agents; ``None`` makes each
    of them choose a uniformly random available action.  The episode ends
    at the first state that is not live or after a collision.
    """
    n = game.num_agents
    if agent < 0 or agent >= n:
        raise ValueError("agent index out of range")
    state = game.initial_state
    r_prev = 0
    out: list[Transition] = []
    while game.is_live(state):
        options = game.options(state)
        profile = [0] * n
        for b in range(n):
            if b == agent:
                continue
            if opponents is None:
                profile[b] = rng.choice(options)
            else:
                j = opponents(state, b)
                if state[j] != UNOWNED:
                    raise ValueError(f"opponent {b} claims occupied sector {j}")
                profile[b] = j
        profile[agent] = epsilon_greedy_action(q, state, options, epsilon, rng)
        prof = tuple(profile)
        if collides(prof):
            r = collision_reward(prof, agent, params)
            q_update(q, game, state, prof[agent], r, None, params)
            out.append(Transition(state, prof, None, r))
            break
        nxt_list = list(state)
        for b, j in enumerate(prof):
            nxt_list[j] = b
        nxt = tuple(nxt_list)
        r_next = game.reward(nxt, agent)
        r = shaped_reward(
            game, state, nxt, agent, params, prev_reward=r_prev, next_reward=r_next
        )
        q_update(q, game, state, prof[agent], r, nxt, params)
        out.append(Transition(state, prof, nxt, r))
        state, r_prev = nxt, r_next
    return tuple(out)


def train_q(
    q: QValues,
    game: LearningGame,
    agent: int,
    *,
    opponents: OpponentFn | None,
    episodes: int,
    params: LearningParams,
    rng: random.Random,
) -> set[StateKey]:
    """Train ``Q_agent`` for *episodes* episodes (one training run of TrainQ).

    Epsilon decreases linearly from ``params.epsilon_start`` in the first
    episode to ``params.epsilon_end`` in the last.  Returns the live states
    visited, at which *agent* chose an action.
    """
    if episodes < 0:
        raise ValueError("episodes must be >= 0")
    visited: set[StateKey] = set()
    for e in range(episodes):
        eps = epsilon_schedule(
            params.epsilon_start, params.epsilon_end, max(1, episodes - 1), e
        )
        for tr in run_episode(
            q, game, agent, opponents=opponents, epsilon=eps, params=params, rng=rng
        ):
            visited.add(tr.state)
    return visited


# ===========================================================================
# Part 2: coordinate-based helpers and the self-play trainer (``rcmas train``)
# ===========================================================================


def state_key(owner_by_index: tuple[int, ...]) -> str:
    """Encode a control vector as a URL-safe base64 string (JSON-friendly)."""
    raw = bytes((o + 1) if o >= 0 else 0 for o in owner_by_index)
    return base64.urlsafe_b64encode(raw).decode("ascii")


def action_key(state: State, action: Coord) -> str:
    """Encode an action as the sector index string."""
    idx = state.territory.index_of(action)
    if idx is None:
        raise ValueError("action not in territory")
    return str(idx)


def decode_action(state: State, ak: str) -> Coord:
    """Decode an action key back to a Coord."""
    return state.ordered_sectors[int(ak)]


@dataclass(slots=True)
class QTable:
    """Simple tabular Q(s,a) storage with JSON serialization."""

    q: dict[str, dict[str, float]]

    @staticmethod
    def empty() -> QTable:
        return QTable(q={})

    def get(self, sk: str, ak: str) -> float:
        return self.q.get(sk, {}).get(ak, 0.0)

    def set(self, sk: str, ak: str, value: float) -> None:
        self.q.setdefault(sk, {})[ak] = float(value)

    def best_action(self, sk: str, action_keys: list[str]) -> str | None:
        return best_q_action(self.q, sk, action_keys)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.q, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load_json(path: Path) -> QTable:
        return QTable(q=json.loads(path.read_text(encoding="utf-8")))


def best_q_action(
    q: dict[str, dict[str, float]],
    sk: str,
    action_keys: list[str],
) -> str | None:
    """The action key with the highest Q-value; the first in *action_keys* on ties."""
    if not action_keys:
        return None
    best = action_keys[0]
    best_v = q.get(sk, {}).get(best, 0.0)
    for ak in action_keys[1:]:
        v = q.get(sk, {}).get(ak, 0.0)
        if v > best_v:
            best, best_v = ak, v
    return best


def top_k_actions(
    q: dict[str, dict[str, float]],
    sk: str,
    action_keys: list[str],
    *,
    k: int,
) -> list[str]:
    """The top-k action keys by Q-value (descending), ties in the order of *action_keys*."""
    if not action_keys or k <= 0:
        return []
    row = q.get(sk, {})
    ranked = sorted(action_keys, key=lambda ak: -row.get(ak, 0.0))
    return ranked[: min(k, len(ranked))]


def _is_live_state(state: State) -> bool:
    """At least ``|Agt|`` unoccupied sectors (Definition 3.8.1)."""
    return sum(1 for o in state.owner_by_index if o == UNOWNED) >= state.num_agents


def compute_shaped_reward(
    prev_state: State,
    next_state: State,
    agent_id: int,
    *,
    agent_acted: bool,
    defeated: bool,
    is_collider: bool,
    territory_size: int,
    collision_penalty: float,
    step_reward: float,
    region_growth_reward: float,
    terminal_reward: float,
    normalize: bool,
) -> float:
    """Shaped reward of Section 4.9 on :class:`~rcmas.model.State` objects.

    A collision (*defeated*) gives ``-collision_penalty`` to the agents that
    caused it and 0 to the others.  Otherwise the reward is *step_reward*
    (if the agent acted) plus the growth of its largest region divided by
    *territory_size*, plus the terminal term if *next_state* is not live.
    """
    if defeated:
        return -collision_penalty if is_collider else 0.0

    norm = float(territory_size) if normalize and territory_size > 0 else 1.0
    r = 0.0

    if agent_acted:
        r += step_reward

    prev_size = largest_region_size(prev_state, agent_id)
    next_size = largest_region_size(next_state, agent_id)
    delta = next_size - prev_size
    if delta:
        r += region_growth_reward * float(delta) / norm

    if not _is_live_state(next_state):
        r += terminal_reward * float(scores(next_state)[agent_id]) / norm

    return r


def compute_truncation_reward(
    state: State,
    agent_id: int,
    territory_size: int,
    *,
    terminal_reward: float,
    normalize: bool,
) -> float:
    """Terminal-like reward when a self-play episode is cut off by ``max_rounds``."""
    norm = float(territory_size) if normalize and territory_size > 0 else 1.0
    return terminal_reward * float(scores(state)[agent_id]) / norm


def detect_collisions(actions: dict[int, Coord | None]) -> frozenset[int]:
    """Return agent IDs whose chosen actions target the same sector."""
    coord_to_agents: dict[Coord, list[int]] = {}
    for aid, act in actions.items():
        if act is not None:
            coord_to_agents.setdefault(act, []).append(aid)
    colliders: set[int] = set()
    for agents in coord_to_agents.values():
        if len(agents) >= 2:
            colliders.update(agents)
    return frozenset(colliders)


def epsilon_schedule(start: float, end: float, decay: int, episode: int) -> float:
    """Linear epsilon decay from *start* (episode 0) to *end* (episode *decay* and later)."""
    if decay <= 0:
        return end
    t = min(1.0, episode / decay)
    return start + t * (end - start)


def _encode_state(
    state: State,
    sym: object | None,
) -> tuple[str, dict[int, int] | None, dict[int, int] | None]:
    """Encode state, optionally canonicalizing under territory symmetry.

    Returns ``(state_key, sigma, inv_sigma)`` where *sigma* maps raw sector
    indices to canonical indices.  When *sym* is ``None``, returns raw
    encoding with ``sigma = inv_sigma = None``.
    """
    if sym is not None:
        from .symmetry import canonical_state, invert_automorphism

        canon_owner, sigma = canonical_state(state.owner_by_index, sym.automorphisms)  # type: ignore[attr-defined]
        return state_key(canon_owner), sigma, invert_automorphism(sigma)
    return state_key(state.owner_by_index), None, None


def _encode_action(
    state: State,
    action: Coord,
    sigma: dict[int, int] | None,
) -> str:
    """Encode action, optionally mapping through automorphism *sigma*."""
    raw = action_key(state, action)
    if sigma is not None:
        return str(sigma[int(raw)])
    return raw


@dataclass(slots=True)
class TrainConfig:
    """Training hyperparameters for self-play Q-learning."""

    episodes: int
    max_rounds: int | None = None
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    # Reward shaping (normalized by territory size by default)
    defeat_penalty: float = 10.0
    step_success_reward: float = 0.1
    delta_region_reward: float = 1.0
    terminal_score_reward: float = 5.0
    normalize_rewards: bool = True
    # Convergence monitoring
    convergence_threshold: float = 0.01
    convergence_window: int = 50
    # Symmetry
    use_symmetry: bool = False


@dataclass(slots=True)
class TrainingArtifacts:
    directory: Path
    qtables: list[QTable]
    converged: bool = False
    final_mean_td_error: float = 0.0
    episodes_completed: int = 0


def train_self_play(
    territory: Territory,
    num_agents: int,
    cfg: TrainConfig,
    *,
    seed: int = 0,
    out_dir: Path | None = None,
) -> TrainingArtifacts:
    """Train independent Q-learners in self-play (``rcmas train``; not used by Q-IBIS).

    All agents learn at once with epsilon-greedy policies.  An episode ends
    at a state that is not live, after a collision, or after *max_rounds*
    rounds (with a terminal-like reward).  Optional symmetry
    canonicalisation of states, and early stopping when the TD error
    stabilises.
    """
    sym = None
    if cfg.use_symmetry:
        from .symmetry import symmetry_info as _sym_info

        sym = _sym_info(territory)

    rngs = [random.Random(seed + i) for i in range(num_agents)]
    qtables = [QTable.empty() for _ in range(num_agents)]
    tsize = len(territory)

    td_history: list[float] = []
    episodes_completed = 0

    for episode in range(cfg.episodes):
        eps = epsilon_schedule(
            cfg.epsilon_start,
            cfg.epsilon_end,
            cfg.epsilon_decay_episodes,
            episode,
        )
        state = State.initial(territory, num_agents)
        rounds = 0
        ep_td_sum = 0.0
        ep_td_count = 0

        while True:
            if not _is_live_state(state):
                break
            if cfg.max_rounds is not None and rounds >= cfg.max_rounds:
                break

            sk, sigma, inv_sigma = _encode_state(state, sym)

            actions: dict[int, Coord | None] = {}
            akeys: dict[int, str] = {}
            for aid in range(num_agents):
                act = _select_action(
                    state,
                    qtables[aid],
                    rngs[aid],
                    eps,
                    sk,
                    sigma,
                    inv_sigma,
                )
                actions[aid] = act
                if act is not None:
                    akeys[aid] = _encode_action(state, act, sigma)

            colliders = detect_collisions(actions)
            if colliders:
                defeated = True
                next_state = state
            else:
                next_state = evolve(state, actions)
                defeated = False

            will_truncate = (
                not defeated
                and _is_live_state(next_state)
                and cfg.max_rounds is not None
                and rounds + 1 >= cfg.max_rounds
            )
            terminal = defeated or not _is_live_state(next_state) or will_truncate

            if not terminal:
                next_sk, next_sigma, _ = _encode_state(next_state, sym)
                next_opts = list(next_state.available_actions())
                next_aks = [
                    _encode_action(next_state, c, next_sigma) for c in next_opts
                ]
            else:
                next_sk = ""
                next_aks = []

            for aid in range(num_agents):
                if aid not in akeys:
                    continue

                r = compute_shaped_reward(
                    prev_state=state,
                    next_state=next_state,
                    agent_id=aid,
                    agent_acted=(actions.get(aid) is not None),
                    defeated=defeated,
                    is_collider=(aid in colliders),
                    territory_size=tsize,
                    collision_penalty=cfg.defeat_penalty,
                    step_reward=cfg.step_success_reward,
                    region_growth_reward=cfg.delta_region_reward,
                    terminal_reward=cfg.terminal_score_reward,
                    normalize=cfg.normalize_rewards,
                )
                if will_truncate:
                    r += compute_truncation_reward(
                        next_state,
                        aid,
                        tsize,
                        terminal_reward=cfg.terminal_score_reward,
                        normalize=cfg.normalize_rewards,
                    )

                if terminal:
                    target = r
                else:
                    best_next = best_q_action(qtables[aid].q, next_sk, next_aks)
                    next_max = (
                        0.0
                        if best_next is None
                        else qtables[aid].q.get(next_sk, {}).get(best_next, 0.0)
                    )
                    target = r + cfg.gamma * next_max

                old = qtables[aid].q.get(sk, {}).get(akeys[aid], 0.0)
                new_val = (1.0 - cfg.alpha) * old + cfg.alpha * float(target)
                qtables[aid].q.setdefault(sk, {})[akeys[aid]] = new_val

                ep_td_sum += abs(target - old)
                ep_td_count += 1

            state = next_state
            rounds += 1
            if terminal:
                break

        episodes_completed += 1
        mean_td = ep_td_sum / max(1, ep_td_count)
        td_history.append(mean_td)

        if cfg.convergence_threshold > 0 and len(td_history) >= cfg.convergence_window:
            window_avg = (
                sum(td_history[-cfg.convergence_window :]) / cfg.convergence_window
            )
            if window_avg < cfg.convergence_threshold:
                break

    final_td = 0.0
    if td_history:
        w = td_history[-min(cfg.convergence_window, len(td_history)) :]
        final_td = sum(w) / len(w)
    converged = cfg.convergence_threshold > 0 and final_td < cfg.convergence_threshold

    if out_dir is None:
        from tempfile import mkdtemp

        out_dir = Path(mkdtemp(prefix="rcmas-qlearning-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, qt in enumerate(qtables):
        qt.save_json(out_dir / f"agent_{i}.json")

    return TrainingArtifacts(
        directory=out_dir,
        qtables=qtables,
        converged=converged,
        final_mean_td_error=final_td,
        episodes_completed=episodes_completed,
    )


def _select_action(
    state: State,
    qtable: QTable,
    rng: random.Random,
    eps: float,
    sk: str,
    sigma: dict[int, int] | None,
    inv_sigma: dict[int, int] | None,
) -> Coord | None:
    """Epsilon-greedy action selection with optional symmetry mapping."""
    options = list(state.available_actions())
    if not options:
        return None
    if rng.random() < eps:
        return rng.choice(options)

    if sigma is not None and inv_sigma is not None:
        canon_aks = [str(sigma[int(action_key(state, c))]) for c in options]
        best = qtable.best_action(sk, canon_aks)
        if best is None:
            return rng.choice(options)
        raw_idx = inv_sigma[int(best)]
        return state.ordered_sectors[raw_idx]
    aks = [action_key(state, c) for c in options]
    best = qtable.best_action(sk, aks)
    return rng.choice(options) if best is None else decode_action(state, best)
