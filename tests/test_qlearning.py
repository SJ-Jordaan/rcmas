"""Tests for qlearning.py (Section 5.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rcmas.model import Coord, State, Territory, evolve, largest_region_size, scores
from rcmas.qlearning import (
    QTable,
    TrainConfig,
    TrainingArtifacts,
    action_key,
    best_q_action,
    compute_shaped_reward,
    compute_truncation_reward,
    decode_action,
    detect_collisions,
    epsilon_schedule,
    state_key,
    top_k_actions,
    train_self_play,
)


# ===================================================================
# State / action encoding
# ===================================================================

class TestStateKey:
    def test_deterministic(self):
        sk1 = state_key((-1, -1, 0, 1))
        sk2 = state_key((-1, -1, 0, 1))
        assert sk1 == sk2

    def test_different_states_differ(self):
        sk1 = state_key((-1, -1))
        sk2 = state_key((0, -1))
        assert sk1 != sk2

    def test_all_unowned(self):
        sk = state_key((-1, -1, -1))
        assert isinstance(sk, str)
        assert len(sk) > 0


class TestActionKey:
    def test_round_trip(self):
        territory = Territory.from_ascii(["..", ".."])
        state = State.initial(territory, num_agents=2)
        coord = Coord(1, 0)
        ak = action_key(state, coord)
        decoded = decode_action(state, ak)
        assert decoded == coord

    def test_all_sectors(self):
        """Every sector encodes to a distinct key."""
        territory = Territory.from_ascii(["..."])
        state = State.initial(territory, num_agents=1)
        keys = [action_key(state, c) for c in territory.ordered_sectors()]
        assert len(keys) == len(set(keys))

    def test_invalid_coord_raises(self):
        territory = Territory.from_ascii([".."])
        state = State.initial(territory, num_agents=1)
        with pytest.raises(ValueError, match="not in territory"):
            action_key(state, Coord(99, 99))


# ===================================================================
# QTable
# ===================================================================

class TestQTable:
    def test_get_set(self):
        qt = QTable.empty()
        assert qt.get("s", "a") == 0.0
        qt.set("s", "a", 1.5)
        assert qt.get("s", "a") == 1.5

    def test_best_action(self):
        qt = QTable.empty()
        qt.set("s", "a0", 1.0)
        qt.set("s", "a1", 5.0)
        qt.set("s", "a2", 3.0)
        assert qt.best_action("s", ["a0", "a1", "a2"]) == "a1"

    def test_best_action_empty_list(self):
        qt = QTable.empty()
        assert qt.best_action("s", []) is None

    def test_best_action_unseen_state(self):
        """All actions have Q=0 for unseen state; returns first."""
        qt = QTable.empty()
        assert qt.best_action("new_state", ["a0", "a1"]) == "a0"

    def test_save_load(self, tmp_path: Path):
        qt = QTable.empty()
        qt.set("s", "a", 1.25)
        p = tmp_path / "q.json"
        qt.save_json(p)
        qt2 = QTable.load_json(p)
        assert qt2.get("s", "a") == 1.25

    def test_save_creates_directory(self, tmp_path: Path):
        qt = QTable.empty()
        qt.set("s", "a", 2.0)
        nested = tmp_path / "sub" / "dir" / "q.json"
        qt.save_json(nested)
        qt2 = QTable.load_json(nested)
        assert qt2.get("s", "a") == 2.0


# ===================================================================
# Q-value helpers
# ===================================================================

class TestQValueHelpers:
    def test_best_q_action(self):
        q: dict[str, dict[str, float]] = {"s": {"a0": 1.0, "a1": 5.0}}
        assert best_q_action(q, "s", ["a0", "a1"]) == "a1"

    def test_best_q_action_empty(self):
        assert best_q_action({}, "s", []) is None

    def test_top_k_actions(self):
        q: dict[str, dict[str, float]] = {"s": {"a0": 1.0, "a1": 5.0, "a2": 3.0}}
        top2 = top_k_actions(q, "s", ["a0", "a1", "a2"], k=2)
        assert top2 == ["a1", "a2"]

    def test_top_k_returns_all_when_k_exceeds(self):
        q: dict[str, dict[str, float]] = {"s": {"a0": 2.0, "a1": 1.0}}
        result = top_k_actions(q, "s", ["a0", "a1"], k=5)
        assert len(result) == 2
        assert result[0] == "a0"

    def test_top_k_zero_k(self):
        q: dict[str, dict[str, float]] = {"s": {"a0": 1.0}}
        assert top_k_actions(q, "s", ["a0"], k=0) == []


# ===================================================================
# compute_shaped_reward
# ===================================================================

class TestComputeShapedReward:
    """Tests for compute_shaped_reward with manual State construction."""

    def _make_state(
        self, territory: Territory, owner_tuple: tuple[int, ...], num_agents: int = 2
    ) -> State:
        """Build a State with a specific ownership vector."""
        return State(
            territory=territory,
            num_agents=num_agents,
            owner_by_index=owner_tuple,
        )

    def test_collision_penalizes_collider(self, territory_2x2: Territory):
        """defeated=True, is_collider=True returns -collision_penalty."""
        s = State.initial(territory_2x2, num_agents=2)
        reward = compute_shaped_reward(
            prev_state=s,
            next_state=s,
            agent_id=0,
            agent_acted=True,
            defeated=True,
            is_collider=True,
            territory_size=len(territory_2x2),
            collision_penalty=10.0,
            step_reward=0.1,
            region_growth_reward=1.0,
            terminal_reward=5.0,
            normalize=True,
        )
        assert reward == -10.0

    def test_collision_spares_bystander(self, territory_2x2: Territory):
        """defeated=True, is_collider=False returns 0.0."""
        s = State.initial(territory_2x2, num_agents=2)
        reward = compute_shaped_reward(
            prev_state=s,
            next_state=s,
            agent_id=0,
            agent_acted=True,
            defeated=True,
            is_collider=False,
            territory_size=len(territory_2x2),
            collision_penalty=10.0,
            step_reward=0.1,
            region_growth_reward=1.0,
            terminal_reward=5.0,
            normalize=True,
        )
        assert reward == 0.0

    def test_step_reward_on_action(self, territory_2x2: Territory):
        """agent_acted=True, not defeated includes step_reward."""
        # prev and next are the same (no region growth, not terminal)
        # so only step_reward contributes
        s = State.initial(territory_2x2, num_agents=2)
        reward = compute_shaped_reward(
            prev_state=s,
            next_state=s,
            agent_id=0,
            agent_acted=True,
            defeated=False,
            is_collider=False,
            territory_size=len(territory_2x2),
            collision_penalty=10.0,
            step_reward=0.5,
            region_growth_reward=1.0,
            terminal_reward=5.0,
            normalize=True,
        )
        # No region growth (both states are empty), not terminal
        assert reward == 0.5

    def test_region_growth_normalized(self, territory_2x2: Territory):
        """Verify delta * region_growth_reward / territory_size."""
        # 2x2 territory: (0,0), (1,0), (0,1), (1,1)
        # prev: agent 0 owns (0,0) -> largest_region_size = 1
        prev = self._make_state(territory_2x2, (0, -1, -1, -1))
        # next: agent 0 owns (0,0) and (1,0) -> largest_region_size = 2
        nxt = self._make_state(territory_2x2, (0, 0, -1, -1))
        reward = compute_shaped_reward(
            prev_state=prev,
            next_state=nxt,
            agent_id=0,
            agent_acted=True,
            defeated=False,
            is_collider=False,
            territory_size=4,
            collision_penalty=10.0,
            step_reward=0.0,
            region_growth_reward=2.0,
            terminal_reward=5.0,
            normalize=True,
        )
        # delta = 2 - 1 = 1, normalized = 2.0 * 1 / 4 = 0.5
        # step_reward = 0, not terminal
        assert reward == pytest.approx(0.5)

    def test_terminal_reward_normalized(self, territory_2x2: Territory):
        """Verify terminal_reward * score / territory_size."""
        # All sectors owned -> terminal.
        # Agent 0 owns (0,0),(1,0) -> score = 2; Agent 1 owns (0,1),(1,1) -> score = 2
        prev = self._make_state(territory_2x2, (0, 0, 1, -1))
        nxt = self._make_state(territory_2x2, (0, 0, 1, 1))
        assert nxt.is_terminal()

        reward = compute_shaped_reward(
            prev_state=prev,
            next_state=nxt,
            agent_id=0,
            agent_acted=False,
            defeated=False,
            is_collider=False,
            territory_size=4,
            collision_penalty=10.0,
            step_reward=0.1,
            region_growth_reward=0.0,
            terminal_reward=5.0,
            normalize=True,
        )
        # agent_acted=False -> no step_reward
        # region_growth_reward=0.0 -> no region component
        # terminal: 5.0 * 2 / 4 = 2.5
        assert reward == pytest.approx(2.5)

    def test_no_normalization(self, territory_2x2: Territory):
        """normalize=False uses raw values (divisor = 1.0)."""
        prev = self._make_state(territory_2x2, (0, 0, 1, -1))
        nxt = self._make_state(territory_2x2, (0, 0, 1, 1))
        assert nxt.is_terminal()

        reward = compute_shaped_reward(
            prev_state=prev,
            next_state=nxt,
            agent_id=0,
            agent_acted=False,
            defeated=False,
            is_collider=False,
            territory_size=4,
            collision_penalty=10.0,
            step_reward=0.1,
            region_growth_reward=0.0,
            terminal_reward=5.0,
            normalize=False,
        )
        # No normalization: terminal = 5.0 * 2 / 1.0 = 10.0
        assert reward == pytest.approx(10.0)


# ===================================================================
# compute_truncation_reward
# ===================================================================

class TestComputeTruncationReward:
    """Tests for compute_truncation_reward."""

    def test_returns_normalized_score(self, territory_2x2: Territory):
        """Verify terminal_reward * score / territory_size."""
        # Agent 0 owns (0,0) and (1,0) -> score = 2
        state = State(
            territory=territory_2x2,
            num_agents=2,
            owner_by_index=(0, 0, 1, -1),
        )
        r = compute_truncation_reward(
            state, agent_id=0, territory_size=4,
            terminal_reward=5.0, normalize=True,
        )
        # 5.0 * 2 / 4 = 2.5
        assert r == pytest.approx(2.5)

    def test_zero_score(self, territory_2x2: Territory):
        """Agent owns nothing -> 0.0."""
        state = State.initial(territory_2x2, num_agents=2)
        r = compute_truncation_reward(
            state, agent_id=0, territory_size=4,
            terminal_reward=5.0, normalize=True,
        )
        assert r == 0.0


# ===================================================================
# detect_collisions
# ===================================================================

class TestDetectCollisions:
    """Tests for detect_collisions."""

    def test_no_collisions(self):
        """Different coords -> empty set."""
        actions = {0: Coord(0, 0), 1: Coord(1, 0)}
        assert detect_collisions(actions) == frozenset()

    def test_two_agents_same_coord(self):
        """Two agents targeting the same sector -> both returned."""
        actions = {0: Coord(0, 0), 1: Coord(0, 0)}
        assert detect_collisions(actions) == frozenset({0, 1})

    def test_three_agents_two_collide(self):
        """Only the agents that collide are returned."""
        actions = {0: Coord(0, 0), 1: Coord(0, 0), 2: Coord(1, 1)}
        colliders = detect_collisions(actions)
        assert colliders == frozenset({0, 1})
        assert 2 not in colliders

    def test_none_actions_ignored(self):
        """None actions (pass) do not cause collisions."""
        actions: dict[int, Coord | None] = {0: None, 1: None}
        assert detect_collisions(actions) == frozenset()

    def test_none_does_not_collide_with_coord(self):
        """A None action does not collide with a real action."""
        actions: dict[int, Coord | None] = {0: Coord(0, 0), 1: None}
        assert detect_collisions(actions) == frozenset()


# ===================================================================
# epsilon_schedule
# ===================================================================

class TestEpsilonSchedule:
    """Tests for linear epsilon decay."""

    def test_start_at_zero(self):
        """episode=0 returns start."""
        assert epsilon_schedule(1.0, 0.05, 500, 0) == 1.0

    def test_end_at_decay(self):
        """episode=decay returns end."""
        assert epsilon_schedule(1.0, 0.05, 500, 500) == pytest.approx(0.05)

    def test_midpoint(self):
        """episode=decay/2 returns (start+end)/2."""
        result = epsilon_schedule(1.0, 0.0, 100, 50)
        assert result == pytest.approx(0.5)

    def test_capped_after_decay(self):
        """episode > decay returns end (capped)."""
        result = epsilon_schedule(1.0, 0.05, 500, 9999)
        assert result == pytest.approx(0.05)

    def test_zero_decay(self):
        """decay=0 always returns end."""
        assert epsilon_schedule(1.0, 0.05, 0, 0) == 0.05
        assert epsilon_schedule(1.0, 0.05, 0, 100) == 0.05


# ===================================================================
# Self-play training
# ===================================================================

class TestTrainSelfPlay:
    def test_runs_and_produces_artifacts(self, tmp_path: Path):
        territory = Territory.from_ascii(["..", ".."])
        cfg = TrainConfig(episodes=5, max_rounds=2)
        artifacts = train_self_play(territory, num_agents=2, cfg=cfg, seed=0, out_dir=tmp_path)
        assert artifacts.directory.exists()
        assert len(artifacts.qtables) == 2
        assert (tmp_path / "agent_0.json").exists()
        assert (tmp_path / "agent_1.json").exists()

    def test_single_agent(self, tmp_path: Path):
        """Single agent self-play should work."""
        territory = Territory.from_ascii([".."])
        cfg = TrainConfig(episodes=3, max_rounds=2)
        artifacts = train_self_play(territory, num_agents=1, cfg=cfg, seed=0, out_dir=tmp_path)
        assert len(artifacts.qtables) == 1

    def test_qtables_have_entries(self, tmp_path: Path):
        """After training, Q-tables should have some entries."""
        territory = Territory.from_ascii(["..", ".."])
        cfg = TrainConfig(episodes=20, max_rounds=2)
        artifacts = train_self_play(territory, num_agents=2, cfg=cfg, seed=42, out_dir=tmp_path)
        for qt in artifacts.qtables:
            assert len(qt.q) > 0


# ===================================================================
# Self-play convergence
# ===================================================================

class TestTrainSelfPlayConvergence:
    """Tests for convergence monitoring fields in TrainingArtifacts."""

    def test_convergence_fields_populated(self, tmp_path: Path):
        """After training, converged/final_mean_td_error/episodes_completed are set."""
        territory = Territory.from_ascii(["..", ".."])
        cfg = TrainConfig(episodes=10, max_rounds=2)
        artifacts = train_self_play(territory, num_agents=2, cfg=cfg, seed=0, out_dir=tmp_path)
        assert isinstance(artifacts.converged, bool)
        assert isinstance(artifacts.final_mean_td_error, float)
        assert artifacts.final_mean_td_error >= 0.0
        assert isinstance(artifacts.episodes_completed, int)
        assert artifacts.episodes_completed > 0
        assert artifacts.episodes_completed <= cfg.episodes

    def test_early_stopping(self, tmp_path: Path):
        """With a high threshold and enough episodes, training stops early."""
        territory = Territory.from_ascii([".."])  # tiny grid: 1 agent fills it fast
        cfg = TrainConfig(
            episodes=500,
            max_rounds=4,
            convergence_threshold=100.0,  # very generous threshold
            convergence_window=3,         # short window to converge quickly
        )
        artifacts = train_self_play(territory, num_agents=1, cfg=cfg, seed=0, out_dir=tmp_path)
        # With such a generous threshold on a tiny grid, early stopping should trigger
        assert artifacts.episodes_completed < cfg.episodes


# ===================================================================
# Self-play with symmetry
# ===================================================================

class TestTrainSelfPlaySymmetry:
    """Tests for symmetry-aware Q-learning training."""

    def test_symmetry_training_runs(self, tmp_path: Path):
        """use_symmetry=True on a 2x2 grid produces valid artifacts."""
        territory = Territory.from_ascii(["..", ".."])
        cfg = TrainConfig(episodes=10, max_rounds=2, use_symmetry=True)
        artifacts = train_self_play(territory, num_agents=2, cfg=cfg, seed=0, out_dir=tmp_path)
        assert artifacts.directory.exists()
        assert len(artifacts.qtables) == 2
        for qt in artifacts.qtables:
            assert isinstance(qt, QTable)

    def test_symmetry_reduces_states(self, tmp_path: Path):
        """Symmetry canonicalization should yield fewer or equal unique state keys."""
        territory = Territory.from_ascii(["..", ".."])

        cfg_no_sym = TrainConfig(episodes=30, max_rounds=4, use_symmetry=False)
        art_no_sym = train_self_play(
            territory, num_agents=2, cfg=cfg_no_sym, seed=42,
            out_dir=tmp_path / "no_sym",
        )

        cfg_sym = TrainConfig(episodes=30, max_rounds=4, use_symmetry=True)
        art_sym = train_self_play(
            territory, num_agents=2, cfg=cfg_sym, seed=42,
            out_dir=tmp_path / "sym",
        )

        # Count unique state keys across all Q-tables
        def total_state_keys(artifacts: TrainingArtifacts) -> int:
            keys: set[str] = set()
            for qt in artifacts.qtables:
                keys.update(qt.q.keys())
            return len(keys)

        states_no_sym = total_state_keys(art_no_sym)
        states_sym = total_state_keys(art_sym)

        # Symmetry canonicalization maps equivalent states together,
        # so the symmetry-aware version should have fewer or equal unique keys
        assert states_sym <= states_no_sym
