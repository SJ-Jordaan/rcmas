from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import Coord


@dataclass(frozen=True, slots=True)
class Territory:
    """A 2D territory of acquirable sectors.

    The territory may be any shape; only coordinates in `sectors` are acquirable.
    """

    sectors: frozenset[Coord]
    _ordered: tuple[Coord, ...]
    _index_by_coord: dict[Coord, int]

    @staticmethod
    def from_ascii(lines: Iterable[str], *, sector: str = ".") -> "Territory":
        rows = [line.rstrip("\n") for line in lines]
        sectors: set[Coord] = set()
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == sector:
                    sectors.add(Coord(x, y))
        ordered = tuple(sorted(sectors, key=lambda c: (c.y, c.x)))
        index_by_coord = {c: i for i, c in enumerate(ordered)}
        return Territory(frozenset(sectors), ordered, index_by_coord)

    def ordered_sectors(self) -> tuple[Coord, ...]:
        return self._ordered

    def index_of(self, coord: Coord) -> int | None:
        return self._index_by_coord.get(coord)

    def __len__(self) -> int:
        return len(self.sectors)
