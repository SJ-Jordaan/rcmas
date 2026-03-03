"""Definition 15: SMT variable construction.

Creates the Z3 integer and boolean variables that encode the RCMAS game
dynamics over a given territory, agent count, and time horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import Coord, Territory, neighbors4


@dataclass(slots=True)
class SmtVariables:
    """All Z3 variables for a single SMT encoding instance.

    Attributes:
        S: number of sectors
        A: number of agents
        T: time horizon (number of rounds)
        sectors: ordered sector coordinates
        owner: ``owner[s][t]`` -- ownership of sector *s* at time *t*
        action: ``action[a][t]`` -- action of agent *a* at time *t*
        adj: ``adj[(i,j,a)]`` -- adjacency indicator for sectors *i < j* owned by *a*
        cr: ``cr[(i,j,a)]``  -- reachability (connected-region) indicator
        size: ``size[i][a]`` -- region size rooted at sector *i* for agent *a*
        payoff: ``payoff[a]`` -- largest-region payoff for agent *a*
        neighbors: physical adjacency map (sector index -> list of neighbour indices)
    """

    S: int
    A: int
    T: int
    sectors: tuple[Coord, ...]
    owner: list[list[Any]]      # [S][T+1]
    action: list[list[Any]]     # [A][T]
    adj: dict[tuple[int, int, int], Any]
    cr: dict[tuple[int, int, int], Any]
    size: list[list[Any]]       # [S][A]
    payoff: list[Any]           # [A]
    neighbors: dict[int, list[int]]
    weights: tuple[int, ...] | None

    def get_adj(self, i: int, j: int, a: int) -> Any:
        lo, hi = (i, j) if i < j else (j, i)
        return self.adj[(lo, hi, a)]

    def get_cr(self, i: int, j: int, a: int) -> Any:
        lo, hi = (i, j) if i < j else (j, i)
        return self.cr[(lo, hi, a)]

    def is_phys_adj(self, i: int, j: int) -> bool:
        return j in self.neighbors[i]


def create_variables(
    territory: Territory,
    num_agents: int,
    horizon: int,
    *,
    weights: tuple[int, ...] | None = None,
    custom_neighbors: dict[int, list[int]] | None = None,
) -> SmtVariables:
    """Construct all Z3 variables for the RCMAS SMT encoding (Def 15).

    When *weights* is provided, ``SmtVariables.weights`` is set and the
    size constraint uses weighted sector contributions.

    When *custom_neighbors* is provided, it replaces the default
    ``neighbors4``-based adjacency.  This is used by the abstract RCMAS
    encoding where sectors are synthetic coordinates and adjacency is
    derived from the concrete territory.
    """
    from z3 import Bool, Int

    sectors = territory.ordered_sectors()
    S = len(sectors)
    A = num_agents
    T = horizon

    owner = [[Int(f"owner_{s}_{t}") for t in range(T + 1)] for s in range(S)]
    action = [[Int(f"action_{a}_{t}") for t in range(T)] for a in range(A)]

    # Physical neighbour map
    if custom_neighbors is not None:
        neighbors = custom_neighbors
    else:
        index_by_coord = {c: i for i, c in enumerate(sectors)}
        neighbors: dict[int, list[int]] = {i: [] for i in range(S)}
        for i, c in enumerate(sectors):
            for nb in neighbors4(c):
                j = index_by_coord.get(nb)
                if j is not None:
                    neighbors[i].append(j)

    # Adjacency and reachability booleans
    adj: dict[tuple[int, int, int], Any] = {}
    cr: dict[tuple[int, int, int], Any] = {}
    for a in range(A):
        for i in range(S):
            for j in range(i + 1, S):
                adj[(i, j, a)] = Bool(f"adj_{i}_{j}_{a}")
                cr[(i, j, a)] = Bool(f"cr_{i}_{j}_{a}")

    # Size and payoff
    size = [[Int(f"size_{i}_{a}") for a in range(A)] for i in range(S)]
    payoff = [Int(f"payoff_{a}") for a in range(A)]

    return SmtVariables(
        S=S, A=A, T=T,
        sectors=sectors,
        owner=owner,
        action=action,
        adj=adj,
        cr=cr,
        size=size,
        payoff=payoff,
        neighbors=neighbors,
        weights=weights,
    )
