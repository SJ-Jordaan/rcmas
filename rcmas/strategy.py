"""Definitions 9-14: Strategy types, reward, outcome, satisfaction.

Defines the strategy-profile representation used by the IBIS and Q-IBIS
algorithms, along with helper functions for freezing/extracting policies
and computing rewards.
"""

from __future__ import annotations

from typing import Iterable

from .model import Coord, State, largest_region_size


# ---------------------------------------------------------------------------
# Def 9-10: Policy representation
# ---------------------------------------------------------------------------

StateKey = tuple[int, ...]
"""Ownership vector used as a hashable state identifier."""

Policy = dict[StateKey, Coord | None]
"""A (partial) deterministic state-only policy: state -> action."""

ActionCandidates = dict[StateKey, tuple[Coord | None, ...]]
"""Per-state candidate action sets (used by Q-IBIS to restrict SMT search)."""


# ---------------------------------------------------------------------------
# Def 11-12: Freezing / hashing helpers
# ---------------------------------------------------------------------------

def freeze_policy(policy: Policy) -> tuple[tuple[StateKey, tuple[int, int] | None], ...]:
    """Return an immutable, hashable snapshot of *policy*."""
    items: list[tuple[StateKey, tuple[int, int] | None]] = []
    for state, action in policy.items():
        items.append((state, None if action is None else (action.x, action.y)))
    items.sort()
    return tuple(items)


def freeze_profile(policies: Iterable[Policy]) -> tuple[tuple[tuple[StateKey, tuple[int, int] | None], ...], ...]:
    """Return an immutable, hashable snapshot of a full policy profile."""
    return tuple(freeze_policy(p) for p in policies)


# ---------------------------------------------------------------------------
# Def 13: Extract policy from an SMT solution trace
# ---------------------------------------------------------------------------

def policy_from_solution(
    owner_by_round: tuple[tuple[int, ...], ...],
    actions_by_round: tuple[tuple[Coord | None, ...], ...],
    agent_id: int,
) -> Policy:
    """Extract agent *agent_id*'s state-only policy from a solved trace."""
    out: Policy = {}
    for t in range(len(actions_by_round)):
        state_key: StateKey = owner_by_round[t]
        out[state_key] = actions_by_round[t][agent_id]
    return out


# ---------------------------------------------------------------------------
# Def 14: Reward / payoff
# ---------------------------------------------------------------------------

def reward(state: State, agent_id: int) -> int:
    """Payoff for *agent_id* in *state*: the size of their largest region (Def 14)."""
    return largest_region_size(state, agent_id)


def fallback_action(state: State, agent_id: int) -> Coord | None:
    """Deterministic collision-free fallback: the *agent_id*-th unowned sector.

    Matches the SMT solver's default_action_expr so that RL and SMT agree
    on the behaviour of unmapped states.
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
