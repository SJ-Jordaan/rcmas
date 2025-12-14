from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GameStatus(str, Enum):
    ONGOING = "ongoing"
    VICTORY = "victory"
    DEFEAT = "defeat"


@dataclass(frozen=True, slots=True)
class GameOutcome:
    status: GameStatus
    reason: str | None = None
