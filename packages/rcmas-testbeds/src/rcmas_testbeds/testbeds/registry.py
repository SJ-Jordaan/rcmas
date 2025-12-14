from __future__ import annotations

from .base import Testbed


def build_testbed(name: str) -> Testbed:
    from .smt_co import SmtCollectiveOptimalityTestbed
    from .qlearning_tb import QLearningTestbed
    from .stubs import HybridTestbed, SmtNaiveNETestbed

    if name == "smt-co":
        return SmtCollectiveOptimalityTestbed()
    if name == "qlearning":
        return QLearningTestbed()
    if name == "smt-ne":
        return SmtNaiveNETestbed()
    if name == "hybrid":
        return HybridTestbed()
    raise ValueError(f"unknown testbed: {name}")
