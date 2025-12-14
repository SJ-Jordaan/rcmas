from pathlib import Path

from qlearning.rl.qtable import QTable


def test_qtable_save_load(tmp_path: Path):
    qt = QTable.empty()
    qt.set("s", "a", 1.25)
    p = tmp_path / "q.json"
    qt.save_json(p)
    qt2 = QTable.load_json(p)
    assert qt2.get("s", "a") == 1.25
