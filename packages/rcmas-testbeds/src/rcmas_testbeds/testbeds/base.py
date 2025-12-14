from __future__ import annotations

from typing import Protocol

from rcmas_core.engine import Territory


class Testbed(Protocol):
    name: str

    def build_agents(self, *, territory: Territory, num_agents: int, max_rounds: int) -> list[object]:
        ...
