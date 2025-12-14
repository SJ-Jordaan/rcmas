from __future__ import annotations

from .base import Testbed


def build_testbed(
    name: str,
    *,
    max_iters: int = 25,
    progress: bool = False,
    timing: bool = False,
    timeout_ms: int | None = None,
) -> Testbed:
    from .smt_co import SmtCollectiveOptimalityTestbed
    from .qlearning_tb import QLearningTestbed
    from .smt_ne import SmtNaiveNETestbed
    from .stubs import HybridTestbed

    if name == "smt-co":
        return SmtCollectiveOptimalityTestbed()
    if name == "qlearning":
        return QLearningTestbed()
    if name == "smt-ne":
        return SmtNaiveNETestbed(max_iters=max_iters, progress=progress, timing=timing, timeout_ms=timeout_ms)
    if name == "hybrid":
        return HybridTestbed()
    raise ValueError(f"unknown testbed: {name}")
