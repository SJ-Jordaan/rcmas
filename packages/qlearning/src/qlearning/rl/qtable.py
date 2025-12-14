from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class QTable:
    """Simple tabular Q(s,a) storage.

    Keys are strings so the table is JSON-serializable.
    """

    q: dict[str, dict[str, float]]

    @staticmethod
    def empty() -> "QTable":
        return QTable(q={})

    def get(self, state_key: str, action_key: str) -> float:
        return self.q.get(state_key, {}).get(action_key, 0.0)

    def set(self, state_key: str, action_key: str, value: float) -> None:
        self.q.setdefault(state_key, {})[action_key] = float(value)

    def best_action(self, state_key: str, action_keys: list[str]) -> str | None:
        if not action_keys:
            return None
        best = action_keys[0]
        best_v = self.get(state_key, best)
        for ak in action_keys[1:]:
            v = self.get(state_key, ak)
            if v > best_v:
                best, best_v = ak, v
        return best

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.q, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load_json(path: Path) -> "QTable":
        return QTable(q=json.loads(path.read_text(encoding="utf-8")))
