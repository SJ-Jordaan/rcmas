"""SMT variables of the encoding of Chapter 4 (Sections 4.2, 4.3 and 4.7).

Creates the Z3 variables of the formula ``F`` and of the opponent
constraint ``[Fix^a_pi]``.  The tool numbers sectors ``0..|T|-1`` (the
enumeration of Remark 3.1.1) and agents ``0..n-1``, so thesis variables map
to tool variables as follows:

=======================  =======================================  ==============================
thesis                   tool                                     values
=======================  =======================================  ==============================
``sec^i_t``              ``owner[i-1][t]``                        ``-1`` for 0, ``a-1`` for ``a``
``act^a_t``              ``action[a-1][t]``                       ``i-1`` for sector ``i``
``in_{i,a}``             ``member[a-1][i-1]``                     Boolean
``root_{i,a}``           ``root[a-1][i-1]``                       Boolean
``pre_{i,a}``            ``prefix[a-1][i-1]``                     Boolean
``lv_{i,a}``             ``level[a-1][i-1]``                      integer
``reward_a``             ``reward[a-1]``                          integer
``unocc^i_t``            ``unocc[t][i-1]``                        integer (Definition 4.7.1)
=======================  =======================================  ==============================

The same variables encode the abstract RCMAS of Definition 6.1.2: its blocks
take the place of the sectors, ``custom_neighbors`` gives the block adjacency
and ``weights`` the block weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import Coord, Territory


@dataclass(slots=True)
class SmtVariables:
    """All Z3 variables for a single SMT encoding instance.

    Attributes:
        S: number of sectors ``|T|``
        A: number of agents ``|Agt|``
        T: horizon ``m`` (number of rounds)
        sectors: ordered sector coordinates
        owner: ``owner[i][t]`` -- control of sector *i* at step *t* (``sec``)
        action: ``action[a][t]`` -- sector claimed by agent *a* at step *t* (``act``)
        member, root, prefix, level: witness variables of Definition 4.3.1
        reward: ``reward[a]`` -- weight of the witness region of agent *a* (Definition 4.3.2)
        unocc: ``unocc[t][i]`` -- number of unoccupied sectors before *i* at step *t*
        neighbors: adjacency map (sector index -> frozenset of neighbour indices)
        weights: sector weights (``None`` for unit weights)
    """

    S: int
    A: int
    T: int
    sectors: tuple[Coord, ...]
    owner: list[list[Any]]  # [S][T+1]
    action: list[list[Any]]  # [A][T]
    member: list[list[Any]]  # [A][S]
    root: list[list[Any]]  # [A][S]
    prefix: list[list[Any]]  # [A][S]
    level: list[list[Any]]  # [A][S]
    reward: list[Any]  # [A]
    unocc: list[list[Any]]  # [T][S]
    neighbors: dict[int, frozenset[int]]
    weights: tuple[int, ...] | None

    @property
    def payoff(self) -> list[Any]:
        """Deprecated alias of :attr:`reward`."""
        return self.reward

    def weight(self, i: int) -> int:
        """Weight of sector *i* (1 for a concrete territory)."""
        return 1 if self.weights is None else self.weights[i]

    def is_phys_adj(self, i: int, j: int) -> bool:
        return j in self.neighbors[i]


def create_variables(
    territory: Territory,
    num_agents: int,
    horizon: int,
    *,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
    max_region_size: int | None = None,
) -> SmtVariables:
    """Construct the Z3 variables of the encoding.

    *custom_neighbors* replaces the four-connected adjacency of the
    territory, and *weights* gives each sector a weight; both are used for
    the abstract RCMAS of Definition 6.1.2.  *max_region_size* is accepted
    for backward compatibility and has no effect: the witness encoding
    needs no pruning.
    """
    from z3 import Bool, Int

    _ = max_region_size
    sectors = territory.ordered_sectors()
    S = len(sectors)
    A = num_agents
    T = horizon
    if weights is not None and len(weights) != S:
        raise ValueError("weights must have one entry per sector")

    owner = [[Int(f"owner_{s}_{t}") for t in range(T + 1)] for s in range(S)]
    action = [[Int(f"action_{a}_{t}") for t in range(T)] for a in range(A)]

    if custom_neighbors is not None:
        neighbors: dict[int, frozenset[int]] = {i: frozenset() for i in range(S)}
        neighbors.update({k: frozenset(v) for k, v in custom_neighbors.items()})
    else:
        neighbors = territory.neighbor_map()

    member = [[Bool(f"in_{i}_{a}") for i in range(S)] for a in range(A)]
    root = [[Bool(f"root_{i}_{a}") for i in range(S)] for a in range(A)]
    prefix = [[Bool(f"pre_{i}_{a}") for i in range(S)] for a in range(A)]
    level = [[Int(f"lv_{i}_{a}") for i in range(S)] for a in range(A)]
    reward = [Int(f"reward_{a}") for a in range(A)]
    unocc = [[Int(f"unocc_{i}_{t}") for i in range(S)] for t in range(T)]

    return SmtVariables(
        S=S,
        A=A,
        T=T,
        sectors=sectors,
        owner=owner,
        action=action,
        member=member,
        root=root,
        prefix=prefix,
        level=level,
        reward=reward,
        unocc=unocc,
        neighbors=neighbors,
        weights=weights,
    )
