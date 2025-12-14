from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, NewType


X = NewType("X", int)
Y = NewType("Y", int)


@dataclass(frozen=True, slots=True)
class Coord:
    x: int
    y: int


def neighbors4(c: Coord) -> Iterable[Coord]:
    yield Coord(c.x + 1, c.y)
    yield Coord(c.x - 1, c.y)
    yield Coord(c.x, c.y + 1)
    yield Coord(c.x, c.y - 1)
