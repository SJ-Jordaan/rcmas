import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

State = Tuple[int, ...]
JointAction = Tuple[int, ...]
QTable = Dict[State, Dict[JointAction, float]]


def q_deltas(prev_table: QTable, new_table: QTable, epsilon: float = 1e-6, limit: int = 20):
    """Yield meaningful Q-value changes up to `limit` entries."""
    changes: List[Dict[str, Any]] = []
    for state, actions in new_table.items():
        baseline_actions = prev_table.get(state, {})
        for joint_action, new_val in actions.items():
            old_val = baseline_actions.get(joint_action, 0.0)
            delta = new_val - old_val
            if abs(delta) > epsilon:
                changes.append(
                    {
                        "state": state,
                        "action": joint_action,
                        "old": old_val,
                        "new": new_val,
                        "delta": delta,
                    }
                )
                if len(changes) >= limit:
                    return changes
    return changes


def top_actions(q_table: QTable, limit: int = 10):
    """Return the top-valued joint actions across all states."""
    top: List[Tuple[float, State, JointAction]] = []
    for state, actions in q_table.items():
        if not actions:
            continue
        best_action, score = max(actions.items(), key=lambda kv: kv[1])
        top.append((score, state, best_action))
    top.sort(key=lambda x: x[0], reverse=True)
    return top[:limit]


def serialize_q_table(q_table: QTable, path: Path, limit_states: int | None = None):
    """Write Q-table to JSON for offline inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {}
    for idx, (state, actions) in enumerate(q_table.items()):
        if limit_states is not None and idx >= limit_states:
            break
        payload[str(state)] = {str(a): v for a, v in actions.items()}

    path.write_text(json.dumps(payload, indent=2))
    return path


def format_changes_table(changes: List[Dict[str, Any]], max_rows: int = 20) -> str:
    """Produce a compact ASCII table for Q-value deltas."""
    header = ["state", "action", "old", "new", "delta"]
    rows = [header]
    for change in changes[:max_rows]:
        rows.append(
            [
                str(change["state"]),
                str(change["action"]),
                f"{change['old']:.3f}",
                f"{change['new']:.3f}",
                f"{change['delta']:.3f}",
            ]
        )

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for row in rows:
        line = " | ".join(text.ljust(col_widths[i]) for i, text in enumerate(row))
        lines.append(line)
    return "\n".join(lines)
