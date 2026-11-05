"""Tests for qlearning.py (thesis Sections 2.4 and 4.9)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import rcmas.qlearning as qlearning
from rcmas.model import Coord, State, Territory
from rcmas.qlearning import (
    LearningGame,
    LearningParams,
    QTable,
    TrainConfig,
    TrainingArtifacts,
    action_key,
    best_q_action,
    collides,
    collision_reward,
    compute_shaped_reward,
    compute_truncation_reward,
    decode_action,
    detect_collisions,
    epsilon_greedy_action,
    epsilon_schedule,
    greedy_action,
    max_q,
    q_update,
    run_episode,
    shaped_reward,
    state_key,
    top_k,
    top_k_actions,
    train_q,
    train_self_play,
)
from rcmas.strategy import CanonicalCompletion, Play, is_live

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
            state,
            agent_id=0,
            territory_size=4,
            terminal_reward=5.0,
            normalize=True,
        )
        # 5.0 * 2 / 4 = 2.5
        assert r == pytest.approx(2.5)

    def test_zero_score(self, territory_2x2: Territory):
        """Agent owns nothing -> 0.0."""
        state = State.initial(territory_2x2, num_agents=2)
        r = compute_truncation_reward(
            state,
            agent_id=0,
            territory_size=4,
            terminal_reward=5.0,
            normalize=True,
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
        artifacts = train_self_play(
            territory, num_agents=2, cfg=cfg, seed=0, out_dir=tmp_path
        )
        assert artifacts.directory.exists()
        assert len(artifacts.qtables) == 2
        assert (tmp_path / "agent_0.json").exists()
        assert (tmp_path / "agent_1.json").exists()

    def test_single_agent(self, tmp_path: Path):
        """Single agent self-play should work."""
        territory = Territory.from_ascii([".."])
        cfg = TrainConfig(episodes=3, max_rounds=2)
        artifacts = train_self_play(
            territory, num_agents=1, cfg=cfg, seed=0, out_dir=tmp_path
        )
        assert len(artifacts.qtables) == 1

    def test_qtables_have_entries(self, tmp_path: Path):
        """After training, Q-tables should have some entries."""
        territory = Territory.from_ascii(["..", ".."])
        cfg = TrainConfig(episodes=20, max_rounds=2)
        artifacts = train_self_play(
            territory, num_agents=2, cfg=cfg, seed=42, out_dir=tmp_path
        )
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
        artifacts = train_self_play(
            territory, num_agents=2, cfg=cfg, seed=0, out_dir=tmp_path
        )
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
            convergence_window=3,  # short window to converge quickly
        )
        artifacts = train_self_play(
            territory, num_agents=1, cfg=cfg, seed=0, out_dir=tmp_path
        )
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
        artifacts = train_self_play(
            territory, num_agents=2, cfg=cfg, seed=0, out_dir=tmp_path
        )
        assert artifacts.directory.exists()
        assert len(artifacts.qtables) == 2
        for qt in artifacts.qtables:
            assert isinstance(qt, QTable)

    def test_symmetry_reduces_states(self, tmp_path: Path):
        """Symmetry canonicalization should yield fewer or equal unique state keys."""
        territory = Territory.from_ascii(["..", ".."])

        cfg_no_sym = TrainConfig(episodes=30, max_rounds=4, use_symmetry=False)
        art_no_sym = train_self_play(
            territory,
            num_agents=2,
            cfg=cfg_no_sym,
            seed=42,
            out_dir=tmp_path / "no_sym",
        )

        cfg_sym = TrainConfig(episodes=30, max_rounds=4, use_symmetry=True)
        art_sym = train_self_play(
            territory,
            num_agents=2,
            cfg=cfg_sym,
            seed=42,
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


# ===================================================================
# Self-play: episodes end at states that are not live
# ===================================================================


class TestSelfPlayLiveness:
    def test_terminal_term_at_non_live_state(self):
        """1x3 with 2 agents: one free sector is left, so the state is terminal."""
        t = Territory.from_ascii(["..."])
        prev = State.initial(t, num_agents=2)
        nxt = State(territory=t, num_agents=2, owner_by_index=(0, 1, -1))
        assert not nxt.is_terminal()
        r = compute_shaped_reward(
            prev_state=prev,
            next_state=nxt,
            agent_id=0,
            agent_acted=True,
            defeated=False,
            is_collider=False,
            territory_size=3,
            collision_penalty=10.0,
            step_reward=0.1,
            region_growth_reward=1.0,
            terminal_reward=5.0,
            normalize=True,
        )
        assert r == pytest.approx(0.1 + 1 / 3 + 5 / 3)

    def test_episode_stops_when_fewer_free_sectors_than_agents(self, tmp_path: Path):
        """No round is played in the state with one free sector (no forced collision)."""
        t = Territory.from_ascii(["..."])
        cfg = TrainConfig(episodes=30, convergence_threshold=0.0)
        art = train_self_play(t, num_agents=2, cfg=cfg, seed=0, out_dir=tmp_path)
        s0 = state_key((-1, -1, -1))
        for qt in art.qtables:
            assert set(qt.q) == {s0}


# ===================================================================
# The learner of Q-IBIS (Section 4.9)
# ===================================================================

P = LearningParams()


def _game_2x2(num_agents: int = 2) -> LearningGame:
    # Sectors 0 1 (top row) and 2 3 (bottom row) in the tool's numbering:
    # 0-1, 0-2, 1-3 and 2-3 are adjacent.
    return LearningGame.from_territory(Territory.from_ascii(["..", ".."]), num_agents)


class TestLearningGame:
    def test_states_and_options(self):
        g = _game_2x2()
        assert g.initial_state == (-1, -1, -1, -1)
        assert g.options((0, -1, 1, -1)) == (1, 3)
        assert g.is_live((0, -1, 1, -1)) and not g.is_live((0, 0, 1, -1))

    def test_rewards_unit_and_weighted(self):
        g = _game_2x2()
        assert g.reward((0, -1, 0, -1), 0) == 2
        assert g.reward((0, 1, 1, 0), 0) == 1
        t = Territory.from_ascii(["...", ""])
        w = LearningGame.from_territory(
            t, 2, weights=(3, 1, 2), custom_neighbors={0: [2], 1: [], 2: [0]}
        )
        assert w.reward((0, 0, 0), 0) == 5  # blocks 0 and 2 adjacent, 1 isolated
        assert w.reward((0, 0, -1), 0) == 3

    def test_weights_must_match(self):
        with pytest.raises(ValueError, match="weights"):
            LearningGame.from_territory(Territory.from_ascii([".."]), 1, weights=(1,))


class TestShapedRewardSection49:
    def test_round_without_collision(self):
        g = _game_2x2()
        s1 = (0, 1, -1, -1)
        assert shaped_reward(g, g.initial_state, s1, 0, P) == pytest.approx(0.1 + 1 / 4)

    def test_terminal_round(self):
        g = _game_2x2()
        s1, s2 = (0, 1, -1, -1), (0, 1, 0, 1)
        expected = 0.1 + (2 - 1) / 4 + 5 * 2 / 4
        assert shaped_reward(g, s1, s2, 0, P) == pytest.approx(expected)

    def test_terminal_when_n_does_not_divide(self):
        g = LearningGame.from_territory(Territory.from_ascii(["..."]), 2)
        r = shaped_reward(g, g.initial_state, (0, 1, -1), 0, P)
        assert r == pytest.approx(0.1 + 1 / 3 + 5 / 3)

    def test_without_normalisation(self):
        g = _game_2x2()
        r = shaped_reward(
            g, (0, 1, -1, -1), (0, 1, 0, 1), 0, LearningParams(normalize=False)
        )
        assert r == pytest.approx(0.1 + 1 + 10)

    def test_collision(self):
        assert collides((0, 0, 1)) and not collides((0, 1, 2))
        assert collision_reward((0, 0, 1), 0, P) == -10.0
        assert collision_reward((0, 0, 1), 1, P) == -10.0
        assert collision_reward((0, 0, 1), 2, P) == 0.0


class TestActionSelection:
    def test_greedy_ties_go_to_smallest_index(self):
        s = (-1,) * 12
        assert greedy_action({}, s, (2, 3, 10)) == 2
        assert greedy_action({s: {3: 1.0, 10: 1.0}}, s, (2, 3, 10)) == 3

    def test_top_k_definition_2_4_4(self):
        s = (-1,) * 12
        q = {s: {10: 1.0, 3: 0.5, 2: 0.5}}
        assert top_k(q, s, (2, 3, 10, 11), 3) == (10, 2, 3)
        assert top_k({}, s, (11, 2, 10), 2) == (2, 10)  # ties by sector index
        assert top_k(q, s, (2, 3), 5) == (2, 3)
        assert top_k(q, s, (2, 3), 0) == ()

    def test_epsilon_greedy(self):
        s = (-1,) * 4
        q = {s: {3: 1.0}}
        rng = random.Random(0)
        assert {
            epsilon_greedy_action(q, s, (0, 1, 2, 3), 0.0, rng) for _ in range(20)
        } == {3}
        assert {
            epsilon_greedy_action(q, s, (0, 1, 2, 3), 1.0, rng) for _ in range(200)
        } == {
            0,
            1,
            2,
            3,
        }


class TestQUpdate:
    def test_equation_2_1(self):
        g = _game_2x2()
        s0, s1 = g.initial_state, (0, 1, -1, -1)
        q = {s1: {2: 2.0, 3: -1.0}}
        new = q_update(q, g, s0, 0, 1.0, s1, P)
        assert new == pytest.approx(0.1 * (1.0 + 0.99 * 2.0))
        assert max_q(q, s1, (2, 3)) == 2.0
        again = q_update(q, g, s0, 0, 1.0, s1, P)
        assert again == pytest.approx(0.9 * new + 0.1 * (1.0 + 0.99 * 2.0))

    def test_terminal_and_failure_have_value_zero(self):
        g = _game_2x2()
        s1, s2 = (0, 1, -1, -1), (0, 1, 0, 1)
        q = {s2: {0: 100.0}}
        assert q_update(q, g, s1, 2, 1.0, s2, P) == pytest.approx(0.1)
        assert q_update(q, g, s1, 3, -10.0, None, P) == pytest.approx(-1.0)

    def test_parameters_are_validated(self):
        with pytest.raises(ValueError, match="learning_rate"):
            LearningParams(learning_rate=0.0)
        with pytest.raises(ValueError, match="epsilon"):
            LearningParams(epsilon_start=1.5)


class TestEpisodes:
    def _play_3x3(self) -> tuple[LearningGame, Play]:
        g = LearningGame.from_territory(Territory.from_ascii(["...", "...", "..."]), 2)
        play = Play.from_actions(9, 2, [[0, 1], [2, 3], [4, 5], [6, 7]])
        return g, play

    def test_online_opponents_follow_the_completion(self):
        g, play = self._play_3x3()
        comp = CanonicalCompletion(play)
        rng = random.Random(1)
        q: dict = {}
        lengths = set()
        for _ in range(50):
            ep = run_episode(
                q, g, 1, opponents=comp.action, epsilon=0.7, params=P, rng=rng
            )
            state = g.initial_state
            for k, tr in enumerate(ep):
                assert tr.state == state and is_live(state, 2)
                assert tr.profile[0] == comp.action(state, 0)
                if tr.next_state is None:
                    assert k == len(ep) - 1 and tr.reward == -10.0
                    break
                assert tr.reward == pytest.approx(
                    shaped_reward(g, state, tr.next_state, 1, P)
                )
                state = tr.next_state
            else:
                assert not g.is_live(state)
            lengths.add(len(ep))
        assert 4 in lengths  # some episodes reach the horizon m = 4

    def test_offline_opponents_are_random(self):
        g = LearningGame.from_territory(Territory.from_ascii(["...", "..."]), 3)
        rng = random.Random(2)
        q: dict = {}
        first = set()
        bystander = False
        for _ in range(200):
            ep = run_episode(q, g, 0, opponents=None, epsilon=0.0, params=P, rng=rng)
            first.add(ep[0].profile[1:])
            last = ep[-1]
            if (
                last.next_state is None
                and last.profile[1] == last.profile[2] != last.profile[0]
            ):
                bystander = True
                assert last.reward == 0.0
        assert len(first) > 10
        assert bystander, "two random opponents collide while the learner does not"

    def test_collision_updates_the_q_value(self):
        g = LearningGame.from_territory(Territory.from_ascii([".."]), 2)
        q: dict = {}
        ep = run_episode(
            q,
            g,
            0,
            opponents=lambda s, b: 0,
            epsilon=0.0,
            params=P,
            rng=random.Random(0),
        )
        assert len(ep) == 1 and ep[0].next_state is None and ep[0].profile == (0, 0)
        assert q[g.initial_state][0] == pytest.approx(-1.0)

    def test_opponent_must_claim_a_free_sector(self):
        g = _game_2x2()
        with pytest.raises(ValueError, match="occupied"):
            run_episode(
                {},
                g,
                0,
                opponents=lambda s, b: 0 if s[0] != -1 else 1,
                epsilon=0.0,
                params=P,
                rng=random.Random(0),
            )


class TestTrainQ:
    def test_linear_epsilon_over_each_run(self, monkeypatch):
        seen = []

        def fake(q, game, agent, *, opponents, epsilon, params, rng):
            seen.append(epsilon)
            return ()

        monkeypatch.setattr(qlearning, "run_episode", fake)
        train_q(
            {},
            _game_2x2(),
            0,
            opponents=None,
            episodes=5,
            params=P,
            rng=random.Random(0),
        )
        assert seen == pytest.approx([0.7, 0.5375, 0.375, 0.2125, 0.05])

    def test_visited_states_are_live_and_reproducible(self):
        g = LearningGame.from_territory(Territory.from_ascii(["...", "...", "..."]), 2)
        q1: dict = {}
        q2: dict = {}
        v1 = train_q(
            q1, g, 0, opponents=None, episodes=40, params=P, rng=random.Random(3)
        )
        v2 = train_q(
            q2, g, 0, opponents=None, episodes=40, params=P, rng=random.Random(3)
        )
        assert v1 == v2 and q1 == q2
        assert g.initial_state in v1
        assert all(is_live(s, 2) for s in v1)
        assert set(q1) == v1
