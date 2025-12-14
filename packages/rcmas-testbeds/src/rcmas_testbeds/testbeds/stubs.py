from __future__ import annotations

from dataclasses import dataclass

from rcmas_core.engine import Territory

from .base import Testbed


@dataclass(frozen=True, slots=True)
class QLearningTestbed(Testbed):
    name: str = "qlearning"

    def build_agents(self, *, territory: Territory, num_agents: int, max_rounds: int) -> list[object]:  # noqa: ARG002
        raise NotImplementedError("Pure Q-learning testbed not implemented yet")


@dataclass(frozen=True, slots=True)
class SmtNaiveNETestbed(Testbed):
    name: str = "smt-ne"

    def build_agents(self, *, territory: Territory, num_agents: int, max_rounds: int) -> list[object]:  # noqa: ARG002
        raise NotImplementedError("Naive NE via SMT not implemented yet")


@dataclass(frozen=True, slots=True)
class HybridTestbed(Testbed):
    name: str = "hybrid"

    def build_agents(self, *, territory: Territory, num_agents: int, max_rounds: int) -> list[object]:  # noqa: ARG002
        raise NotImplementedError("Hybrid SMT+Q-learning not implemented yet")
