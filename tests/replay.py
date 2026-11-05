"""Replaying the answers of the SMT queries of one run in another run.

The tests of :func:`rcmas.ibis.at_reward_bound` compare a run that skips the
best-response query of an agent whose reward is ``R_max`` with a run that
poses it.  Z3's models depend on process-wide state (the terms created by
earlier queries among it), so two runs that pose different queries may get
different models of the same query, and a direct comparison of their plays
would test Z3 rather than the skip.  :class:`QueryReplay` therefore records
every query of the run that poses all of them and answers the other run's
queries from that record.  A query that the first run never posed fails the
test: with the skip, a run must pose a subset of the queries of the run
without it, and get the same answers, so that it takes the same path.

The queries are those of IBIS and S-IBIS (initial query, ``smt(G_a(pi),
reward_a)``, the S-IBIS filter and binary search), of Q-IBIS (the
restricted query) and of VerifyNE in CEGAR-NE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import rcmas.cegar as cegar_mod
import rcmas.ibis as ibis_mod
import rcmas.qibis as qibis_mod


def disable_skip(monkeypatch: Any) -> None:
    """Pose every best-response and VerifyNE query, also for agents at ``R_max``."""
    never: Callable[..., bool] = lambda base, current, agent: False  # noqa: E731
    for mod in (ibis_mod, qibis_mod, cegar_mod):
        monkeypatch.setattr(mod, "at_reward_bound", never)


def _game(base: Any) -> tuple:
    """The game of a formula ``F``: agents, horizon, adjacency and weights."""
    v = base.variables
    nbrs = tuple(sorted((k, tuple(sorted(x))) for k, x in v.neighbors.items()))
    return (v.A, v.T, v.S, nbrs, v.weights, base.demands)


def _cands(cands: Any) -> Any:
    if cands is None:
        return None
    return tuple(
        None if c is None else tuple(sorted((k, tuple(x)) for k, x in c.items()))
        for c in cands
    )


@dataclass
class QueryReplay:
    """Record the queries of one run, then answer those of another from the record.

    Attributes:
        answers: query key -> answer, filled by the recording run.
        posed: query keys in the order the current run posed them.
        mode: ``"record"`` or ``"replay"``.
    """

    monkeypatch: Any
    answers: dict[tuple, Any] = field(default_factory=dict)
    posed: list[tuple] = field(default_factory=list)
    mode: str = "record"

    def __post_init__(self) -> None:
        real_init = ibis_mod.initial_from_base
        real_br = ibis_mod.best_response_from_base
        real_bs = ibis_mod.best_response_binsearch_from_base
        real_rbr = qibis_mod.best_response_from_base
        real_ver = cegar_mod.check_improvement_from_base

        def init(base: Any, **kw: Any) -> Any:
            key = ("initial", _game(base), kw.get("initial"), kw.get("symmetry"))
            return self._answer(key, lambda: real_init(base, **kw))

        def br(base: Any, play: Any, agent: int, **kw: Any) -> Any:
            key = ("br", _game(base), play, agent)
            return self._answer(key, lambda: real_br(base, play, agent, **kw))

        def bs(base: Any, play: Any, agent: int, **kw: Any) -> Any:
            key = ("bs", _game(base), play, agent, kw.get("current_reward"))
            return self._answer(key, lambda: real_bs(base, play, agent, **kw))

        def rbr(base: Any, play: Any, agent: int, **kw: Any) -> Any:
            cands = _cands(kw.get("action_candidates_by_agent"))
            key = ("restricted", _game(base), play, agent, cands)
            return self._answer(key, lambda: real_rbr(base, play, agent, **kw))

        def ver(base: Any, **kw: Any) -> Any:
            key = (
                "verify",
                _game(base),
                kw.get("opponent_play"),
                kw.get("agent"),
                kw.get("threshold"),
            )
            return self._answer(key, lambda: real_ver(base, **kw))

        mp = self.monkeypatch
        mp.setattr(ibis_mod, "initial_from_base", init)
        mp.setattr(ibis_mod, "best_response_from_base", br)
        mp.setattr(ibis_mod, "best_response_binsearch_from_base", bs)
        mp.setattr(qibis_mod, "best_response_from_base", rbr)
        mp.setattr(cegar_mod, "check_improvement_from_base", ver)

    def _answer(self, key: tuple, ask: Callable[[], Any]) -> Any:
        self.posed.append(key)
        if self.mode == "record":
            if key not in self.answers:
                self.answers[key] = ask()
            return self.answers[key]
        assert (
            key in self.answers
        ), f"query not posed by the run without the skip: {key[0]}"
        return self.answers[key]

    def replay(self) -> list[tuple]:
        """Switch to replay; returns the queries of the recording run."""
        recorded, self.posed, self.mode = self.posed, [], "replay"
        return recorded

    def best_responses(self, posed: list[tuple] | None = None) -> list[tuple]:
        """The best-response and VerifyNE queries among *posed* (default: the current run)."""
        keys = self.posed if posed is None else posed
        return [k for k in keys if k[0] != "initial"]
