"""Shared core package for the RCMAS territory game."""

from .engine import Coord, GameEngine, GameOutcome, GameResult, GameState, GameStatus, Territory

__all__ = [
    "Coord",
    "Territory",
    "GameState",
    "GameEngine",
    "GameResult",
    "GameOutcome",
    "GameStatus",
]
