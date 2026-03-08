"""Section 5.2: Q-learning for RCMAS.

Provides tabular Q-learning with state/action encoding, a Q-table,
shaped reward computation, and training functions used by Q-IBIS.
Also includes the self-play trainer for standalone Q-learning experiments.

Design principles:
- Rewards normalized by territory size for hyperparameter transferability
- Per-agent collision credit (only colliding agents are penalized)
- Truncation-safe: terminal-like reward computed at max_rounds cutoff
- Optional symmetry canonicalization in state encoding
- Convergence monitoring via TD-error tracking with early stopping
"""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .model import Coord, State, Territory, CollisionError, evolve, largest_region_size, scores


# ---------------------------------------------------------------------------
# State / action encoding
# ---------------------------------------------------------------------------

def state_key(owner_by_index: tuple[int, ...]) -> str:
    """Encode an ownership vector as a URL-safe base64 string."""
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


# ---------------------------------------------------------------------------
# Q-table
# ---------------------------------------------------------------------------

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
        if not action_keys:
            return None
        best = action_keys[0]
        best_v = self.get(sk, best)
        for ak in action_keys[1:]:
            v = self.get(sk, ak)
            if v > best_v:
                best, best_v = ak, v
        return best

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.q, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load_json(path: Path) -> QTable:
        return QTable(q=json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Q-value helpers
# ---------------------------------------------------------------------------

def best_q_action(
    q: dict[str, dict[str, float]], sk: str, action_keys: list[str],
) -> str | None:
    """Return the action key with the highest Q-value."""
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
    q: dict[str, dict[str, float]], sk: str, action_keys: list[str], *, k: int,
) -> list[str]:
    """Return the top-k action keys by Q-value (descending)."""
    if not action_keys or k <= 0:
        return []
    scored = [(q.get(sk, {}).get(ak, 0.0), ak) for ak in action_keys]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [ak for _, ak in scored[:min(k, len(scored))]]


# ---------------------------------------------------------------------------
# Shared reward shaping (used by both self-play and Q-IBIS)
# ---------------------------------------------------------------------------

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
    """Compute shaped reward with normalization and per-agent collision credit.

    When *normalize* is True, region-size-dependent reward components are
    divided by *territory_size* so that hyperparameters transfer across
    different grid sizes.

    Only agents that actually caused a collision receive the collision
    penalty; bystanders receive 0.
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

    if next_state.is_terminal():
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
    """Compute terminal-like reward when episode is truncated by max_rounds.

    Ensures Q-values reflect the game objective even when episodes end
    before every sector is claimed.
    """
    norm = float(territory_size) if normalize and territory_size > 0 else 1.0
    return terminal_reward * float(scores(state)[agent_id]) / norm


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Epsilon schedule
# ---------------------------------------------------------------------------

def epsilon_schedule(start: float, end: float, decay: int, episode: int) -> float:
    """Linear epsilon decay from *start* to *end* over *decay* episodes."""
    if decay <= 0:
        return end
    t = min(1.0, episode / decay)
    return start + t * (end - start)


# ---------------------------------------------------------------------------
# Canonical state helpers (for symmetry-aware Q-learning)
# ---------------------------------------------------------------------------

def _encode_state(
    state: State, sym: object | None,
) -> tuple[str, dict[int, int] | None, dict[int, int] | None]:
    """Encode state, optionally canonicalizing under territory symmetry.

    Returns ``(state_key, sigma, inv_sigma)`` where *sigma* maps raw sector
    indices to canonical indices.  When *sym* is ``None``, returns raw
    encoding with ``sigma = inv_sigma = None``.
    """
    if sym is not None:
        from .symmetry import canonical_state, invert_automorphism
        canon_owner, sigma = canonical_state(state.owner_by_index, sym.automorphisms)
        return state_key(canon_owner), sigma, invert_automorphism(sigma)
    return state_key(state.owner_by_index), None, None


def _encode_action(
    state: State, action: Coord, sigma: dict[int, int] | None,
) -> str:
    """Encode action, optionally mapping through automorphism *sigma*."""
    raw = action_key(state, action)
    if sigma is not None:
        return str(sigma[int(raw)])
    return raw


# ---------------------------------------------------------------------------
# Self-play trainer
# ---------------------------------------------------------------------------

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
    """Train independent Q-learners in self-play.

    Improvements over plain IQL:
    - Truncation-safe: computes terminal-like reward at *max_rounds* cutoff
    - Per-agent collision credit: only colliding agents are penalized
    - Normalized rewards: scale with territory size for transferability
    - Optional symmetry canonicalization reduces effective state space
    - Convergence monitoring: stops early when TD-error stabilizes
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
            cfg.epsilon_start, cfg.epsilon_end,
            cfg.epsilon_decay_episodes, episode,
        )
        state = State.initial(territory, num_agents)
        rounds = 0
        ep_td_sum = 0.0
        ep_td_count = 0

        while True:
            if state.is_terminal():
                break
            if cfg.max_rounds is not None and rounds >= cfg.max_rounds:
                break

            # Canonical state (computed once per step, shared across agents)
            sk, sigma, inv_sigma = _encode_state(state, sym)

            # Select actions for all agents
            actions: dict[int, Coord | None] = {}
            akeys: dict[int, str] = {}
            for aid in range(num_agents):
                act = _select_action(
                    state, qtables[aid], rngs[aid], eps,
                    sk, sigma, inv_sigma,
                )
                actions[aid] = act
                if act is not None:
                    akeys[aid] = _encode_action(state, act, sigma)

            # Detect collisions before calling evolve
            colliders = detect_collisions(actions)
            if colliders:
                defeated = True
                next_state = state
            else:
                next_state = evolve(state, actions)
                defeated = False

            # Will the *next* round be truncated?
            will_truncate = (
                not defeated
                and not next_state.is_terminal()
                and cfg.max_rounds is not None
                and rounds + 1 >= cfg.max_rounds
            )
            terminal = defeated or next_state.is_terminal() or will_truncate

            # Next-state encoding (computed once for Q-update bootstrap)
            if not terminal:
                next_sk, next_sigma, _ = _encode_state(next_state, sym)
                next_opts = list(next_state.available_actions())
                next_aks = [_encode_action(next_state, c, next_sigma) for c in next_opts]
            else:
                next_sk = None
                next_aks = []

            # Per-agent reward computation and Q-update
            for aid in range(num_agents):
                if aid not in akeys:
                    continue

                r = compute_shaped_reward(
                    prev_state=state, next_state=next_state, agent_id=aid,
                    agent_acted=(actions.get(aid) is not None),
                    defeated=defeated, is_collider=(aid in colliders),
                    territory_size=tsize,
                    collision_penalty=cfg.defeat_penalty,
                    step_reward=cfg.step_success_reward,
                    region_growth_reward=cfg.delta_region_reward,
                    terminal_reward=cfg.terminal_score_reward,
                    normalize=cfg.normalize_rewards,
                )
                if will_truncate:
                    r += compute_truncation_reward(
                        next_state, aid, tsize,
                        terminal_reward=cfg.terminal_score_reward,
                        normalize=cfg.normalize_rewards,
                    )

                # Bellman Q-update
                if terminal:
                    target = r
                else:
                    best_next = best_q_action(qtables[aid].q, next_sk, next_aks)
                    next_max = (
                        0.0 if best_next is None
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

        # Early stopping on convergence
        if (
            cfg.convergence_threshold > 0
            and len(td_history) >= cfg.convergence_window
        ):
            window_avg = (
                sum(td_history[-cfg.convergence_window:])
                / cfg.convergence_window
            )
            if window_avg < cfg.convergence_threshold:
                break

    # Final convergence stats
    final_td = 0.0
    if td_history:
        w = td_history[-min(cfg.convergence_window, len(td_history)):]
        final_td = sum(w) / len(w)
    converged = cfg.convergence_threshold > 0 and final_td < cfg.convergence_threshold

    # Save artifacts
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


# ---------------------------------------------------------------------------
# Internal action selection
# ---------------------------------------------------------------------------

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
    else:
        aks = [action_key(state, c) for c in options]
        best = qtable.best_action(sk, aks)
        return rng.choice(options) if best is None else decode_action(state, best)
