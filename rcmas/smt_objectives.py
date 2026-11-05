"""Objective functions of the encoding (thesis Sections 4.6 and 4.7).

- :func:`collective_objective`: maximise ``f = sum_a reward_a``, the
  quantitative objective function of Definition 4.6.1, for collectively
  optimal synthesis (Theorem 4.6.1).
- :func:`individual_objective`: maximise ``reward_a``, the objective of the
  best-response query ``smt(G_a(pi), reward_a)`` (Definition 4.7.1).

The qualitative objective, ``reward_a >= d(a)``, is a constraint of ``F``
(:func:`rcmas.smt_constraints.qualitative_constraint`).  The witness encoding
is exact for maximisation (Corollary 4.3.3), so both objectives reach the
true rewards of Definition 3.9.2 at the optimum.
"""

from __future__ import annotations

from typing import Any

from .smt_variables import SmtVariables


def collective_objective(opt: Any, v: SmtVariables) -> None:
    """Maximise the sum of all agents' rewards (Definition 4.6.1)."""
    from z3 import IntVal, Sum

    opt.maximize(Sum(v.reward) if v.reward else IntVal(0))


def individual_objective(opt: Any, v: SmtVariables, agent_id: int) -> None:
    """Maximise agent *agent_id*'s reward (the best-response query, Definition 4.7.1)."""
    if agent_id < 0 or agent_id >= v.A:
        raise ValueError("objective agent index out of range")
    opt.maximize(v.reward[agent_id])
