"""Definitions 24-25: SMT objective functions.

Provides the two objective modes used in the paper:
- Collective optimality (maximise sum of payoffs)
- Individual best-response (maximise a single agent's payoff)
"""

from __future__ import annotations

from typing import Any

from .smt_variables import SmtVariables


# ---------------------------------------------------------------------------
# Def 24: Qualitative objective (collective optimality)
# ---------------------------------------------------------------------------

def qualitative_objective(opt: Any, v: SmtVariables) -> None:
    """Maximise the sum of all agents' payoffs."""
    from z3 import Sum

    opt.maximize(Sum(v.payoff))


# ---------------------------------------------------------------------------
# Def 25: Quantitative objective (individual best-response)
# ---------------------------------------------------------------------------

def quantitative_objective(opt: Any, v: SmtVariables, agent_id: int) -> None:
    """Maximise agent *agent_id*'s payoff."""
    if agent_id < 0 or agent_id >= v.A:
        raise ValueError("objective agent index out of range")
    opt.maximize(v.payoff[agent_id])
