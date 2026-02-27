"""Tests for qlearning.py (Section 5.2)."""

from __future__ import annotations

from pathlib import Path

from rcmas.model import Coord, State, Territory
from rcmas.qlearning import (
    QTable,
    TrainConfig,
    action_key,
    best_q_action,
    decode_action,
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
        import pytest
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
