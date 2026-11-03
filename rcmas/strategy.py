"""Strategies, plays and the canonical completion (thesis Sections 3.8 and 4.4).

A strategy (Definition 3.8.1) maps every live state, a control map with at
least ``|Agt|`` unoccupied sectors, to an action.  The synthesisers produce a
play (Definition 4.2.5), and the strategies they return are its canonical
completion (Definition 4.4.2): at a state of the play every agent takes its
recorded action, and at every other live state agent ``a`` claims the
``a``-th unoccupied sector in the enumeration of Remark 3.1.1.

Indices follow :mod:`rcmas.model`: sectors ``0..|T|-1`` in the enumeration
order, agents ``0..n-1``, and ``UNOWNED`` (``-1``) for an unoccupied sector.
Thesis agent ``a`` is tool agent ``a-1``, so in the tool agent ``k`` claims
the unoccupied sector with exactly ``k`` unoccupied sectors before it.

The dictionary-based :data:`Policy` helpers at the end of the module are kept
for Q-learning and for callers that describe strategies by coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .model import (
    UNOWNED,
    CollisionError,
    Coord,
    State,
    Territory,
    largest_region_size,
    rewards_from_owner,
)

StateKey = tuple[int, ...]
"""Control vector used as a hashable state identifier (``UNOWNED`` = -1)."""

Policy = dict[StateKey, Coord | None]
"""A (partial) deterministic state-only policy: state -> action."""

ActionCandidates = dict[StateKey, tuple[Coord | None, ...]]
"""Per-state candidate action sets (used by Q-IBIS to restrict SMT search)."""


# ---------------------------------------------------------------------------
# Live states and default actions (Definitions 3.8.1 and 4.4.2)
# ---------------------------------------------------------------------------


def num_unoccupied(state: StateKey) -> int:
    """Number of unoccupied sectors of a control vector."""
    return sum(1 for o in state if o == UNOWNED)


def is_live(state: StateKey, num_agents: int) -> bool:
    """True if at least ``num_agents`` sectors are unoccupied (Definition 3.8.1)."""
    return num_unoccupied(state) >= num_agents


def free_sector(state: StateKey, agent: int) -> int | None:
    """Index of ``free_a(s)``, the unoccupied sector with *agent* unoccupied sectors before it.

    Tool agent *agent* is thesis agent ``agent + 1``, whose ``free`` sector is
    the ``(agent + 1)``-th unoccupied sector (Definition 4.4.2).  Returns
    ``None`` if fewer than ``agent + 1`` sectors are unoccupied.
    """
    if agent < 0:
        raise ValueError("agent index must be >= 0")
    seen = 0
    for i, o in enumerate(state):
        if o == UNOWNED:
            if seen == agent:
                return i
            seen += 1
    return None


def default_action(state: StateKey, agent: int, num_agents: int) -> int:
    """Sector claimed by the default action ``def_a(s)`` of Definition 4.4.2.

    Defined only at live states, where it exists for every agent and two
    agents never claim the same sector (Lemma 4.4.1).
    """
    if agent < 0 or agent >= num_agents:
        raise ValueError("agent index out of range")
    if not is_live(state, num_agents):
        raise ValueError("the default action is defined only at live states")
    idx = free_sector(state, agent)
    assert idx is not None
    return idx


# ---------------------------------------------------------------------------
# Plays (Definition 4.2.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Play:
    """A play ``s_0 ap_0 s_1 ... ap_{m-1} s_m`` of an RCMAS (Definition 4.2.5).

    Attributes:
        states: the control vectors ``s_0, ..., s_m`` (``m + 1`` entries).
        actions: the profiles ``ap_0, ..., ap_{m-1}``; ``actions[t][a]`` is
            the index of the sector that agent ``a`` claims in round ``t``.
        num_agents: ``|Agt|``.

    Plays are hashable, so a run can keep its plays in a set (line 6 of
    Algorithm 1).  Use :meth:`from_actions` to build a play and check it.
    """

    states: tuple[StateKey, ...]
    actions: tuple[tuple[int, ...], ...]
    num_agents: int

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("a play has one more state than profiles")
        for prof in self.actions:
            if len(prof) != self.num_agents:
                raise ValueError("every profile needs one action per agent")

    @staticmethod
    def from_actions(
        num_sectors: int,
        num_agents: int,
        actions: Iterable[Iterable[int]],
    ) -> Play:
        """Simulate *actions* from ``s_0`` and return the play.

        Raises :class:`~rcmas.model.CollisionError` if a profile collides and
        :class:`ValueError` if it claims an occupied or unknown sector, so the
        result is a play in the sense of Definition 4.2.5.
        """
        state = [UNOWNED] * num_sectors
        states: list[StateKey] = [tuple(state)]
        profiles: list[tuple[int, ...]] = []
        for raw in actions:
            prof = tuple(int(i) for i in raw)
            if len(prof) != num_agents:
                raise ValueError("every profile needs one action per agent")
            if len(set(prof)) != len(prof):
                raise CollisionError(f"collision in round {len(profiles)}: {prof}")
            for a, i in enumerate(prof):
                if i < 0 or i >= num_sectors:
                    raise ValueError(f"sector index {i} out of range")
                if state[i] != UNOWNED:
                    raise ValueError(f"agent {a} claims occupied sector {i}")
            for a, i in enumerate(prof):
                state[i] = a
            profiles.append(prof)
            states.append(tuple(state))
        return Play(tuple(states), tuple(profiles), num_agents)

    @staticmethod
    def from_coord_actions(
        territory: Territory,
        num_agents: int,
        actions_by_round: Iterable[Iterable[Coord | None]],
    ) -> Play:
        """Build a play from per-round coordinate actions (as in ``SmtSolution``)."""
        idx_actions: list[list[int]] = []
        for step in actions_by_round:
            row: list[int] = []
            for c in step:
                if c is None:
                    raise ValueError("a play has no pass actions")
                i = territory.index_of(c)
                if i is None:
                    raise ValueError(f"action {c} is not in the territory")
                row.append(i)
            idx_actions.append(row)
        return Play.from_actions(len(territory), num_agents, idx_actions)

    @property
    def horizon(self) -> int:
        """Number of rounds ``m``."""
        return len(self.actions)

    @property
    def final_state(self) -> StateKey:
        """The final control vector ``s_m``."""
        return self.states[-1]

    def step_of(self, state: StateKey) -> int | None:
        """The step ``t < m`` with ``s_t = state``, or ``None`` if *state* is off the play."""
        for t in range(self.horizon):
            if self.states[t] == state:
                return t
        return None

    def claims_of(self, agent: int) -> tuple[int, ...]:
        """The sectors that *agent* claims, round by round."""
        return tuple(prof[agent] for prof in self.actions)

    def to_coords(self, territory: Territory) -> tuple[tuple[Coord, ...], ...]:
        """The profiles as coordinates, in the format of ``SmtSolution.actions_by_round``."""
        sectors = territory.ordered_sectors()
        return tuple(tuple(sectors[i] for i in prof) for prof in self.actions)

    def rewards(
        self,
        neighbors: dict[int, frozenset[int]] | dict[int, list[int]],
        weights: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """``r_a(pi)`` for every agent, evaluated on the final state (Definition 3.9.2)."""
        return rewards_from_owner(self.final_state, self.num_agents, neighbors, weights)


# ---------------------------------------------------------------------------
# Canonical completion (Definition 4.4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalCompletion:
    """The canonical completion of a play (Definition 4.4.2).

    ``action(s, a)`` is the recorded action ``ap_t(a)`` if ``s = s_t`` for
    some ``t < m``, and the default action ``def_a(s)`` at every other live
    state.  It is undefined (``ValueError``) at states that are not live.
    """

    play: Play
    _step: dict[StateKey, int] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        step = {self.play.states[t]: t for t in range(self.play.horizon)}
        if len(step) != self.play.horizon:  # pragma: no cover - Lemma 4.4.1(3)
            raise ValueError("the states of a play are pairwise distinct")
        object.__setattr__(self, "_step", step)

    @property
    def num_agents(self) -> int:
        return self.play.num_agents

    def is_on_play(self, state: StateKey) -> bool:
        """True if *state* is one of ``s_0, ..., s_{m-1}``."""
        return state in self._step

    def action(self, state: StateKey, agent: int) -> int:
        """Sector index of ``alpha_a(s)`` under the canonical completion."""
        t = self._step.get(state)
        if t is not None:
            return self.play.actions[t][agent]
        return default_action(state, agent, self.play.num_agents)

    def joint_action(self, state: StateKey) -> tuple[int, ...]:
        """The profile ``(alpha_1(s), ..., alpha_n(s))``."""
        return tuple(self.action(state, a) for a in range(self.play.num_agents))

    def policy(self, territory: Territory, agent: int) -> Policy:
        """The recorded actions of *agent* as a coordinate :data:`Policy`.

        With :func:`~rcmas.smt_constraints.fixed_policy_constraint`, whose
        unmapped states take the default action, this policy encodes the
        canonical completion.
        """
        sectors = territory.ordered_sectors()
        return {
            self.play.states[t]: sectors[self.play.actions[t][agent]]
            for t in range(self.play.horizon)
        }


def canonical_completion(play: Play) -> CanonicalCompletion:
    """Return the canonical completion of *play* (Definition 4.4.2)."""
    return CanonicalCompletion(play)


# ---------------------------------------------------------------------------
# Dictionary policies (used by Q-learning and Q-IBIS)
# ---------------------------------------------------------------------------


def freeze_policy(
    policy: Policy,
) -> tuple[tuple[StateKey, tuple[int, int] | None], ...]:
    """Return an immutable, hashable snapshot of *policy*."""
    items: list[tuple[StateKey, tuple[int, int] | None]] = []
    for state, action in policy.items():
        items.append((state, None if action is None else (action.x, action.y)))
    items.sort()
    return tuple(items)


def freeze_profile(
    policies: Iterable[Policy],
) -> tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]:
    """Return an immutable, hashable snapshot of a full policy profile."""
    return tuple(freeze_policy(p) for p in policies)


def policy_from_solution(
    owner_by_round: tuple[tuple[int, ...], ...],
    actions_by_round: tuple[tuple[Coord | None, ...], ...],
    agent_id: int,
) -> Policy:
    """Strategy of *agent_id* derived from a solved trace (Definition 4.4.1)."""
    out: Policy = {}
    for t in range(len(actions_by_round)):
        state_key: StateKey = owner_by_round[t]
        out[state_key] = actions_by_round[t][agent_id]
    return out


def reward(state: State, agent_id: int) -> int:
    """Reward of *agent_id* in *state*: the size of its largest cohesive region (Definition 3.9.2)."""
    return largest_region_size(state, agent_id)


def fallback_action(state: State, agent_id: int) -> Coord | None:
    """Default action of *agent_id*: the ``(agent_id + 1)``-th unoccupied sector.

    This is ``def_a`` of Definition 4.4.2 at live states, and matches the
    default action that :func:`~rcmas.smt_constraints.fixed_policy_constraint`
    and :func:`~rcmas.smt_constraints.opponent_constraint` encode.  Returns
    ``None`` if fewer than ``agent_id + 1`` sectors are unoccupied.
    """
    options = state.available_actions()
    if agent_id < 0 or agent_id >= state.num_agents:
        raise ValueError("agent_id out of range")
    if agent_id >= len(options):
        return None
    return options[agent_id]


def policy_action(policy: Policy, state: State, agent_id: int) -> Coord | None:
    """Look up *policy* for the current state, falling back to :func:`fallback_action`."""
    key: StateKey = tuple(state.owner_by_index)
    chosen = policy.get(key)
    if chosen is None:
        return fallback_action(state, agent_id)
    if chosen not in state.available_actions():
        return fallback_action(state, agent_id)
    return chosen
