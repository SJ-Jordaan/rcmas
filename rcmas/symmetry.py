"""Symmetry detection and reduction for RCMAS territories.

Computes the automorphism group of a territory (a subgroup of D_4 for
square grids, V_4 for rectangles), partitions sectors into orbits under
this group, and selects canonical orbit representatives for use in
symmetry-breaking SMT constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Coord, Territory, neighbors4


@dataclass(frozen=True, slots=True)
class SymmetryInfo:
    """Bundle of symmetry data for a territory.

    Attributes:
        automorphisms: list of sector-index permutations (as dicts)
                       that preserve the territory and its adjacency.
        orbits: partition of sector indices into equivalence classes
                under the automorphism group.
        representatives: one canonical (smallest-index) representative
                         per orbit.
    """

    automorphisms: tuple[dict[int, int], ...]
    orbits: tuple[frozenset[int], ...]
    representatives: tuple[int, ...]


def _bounding_box(
    territory: Territory,
) -> tuple[int, int, int, int]:
    """Return (min_x, min_y, max_x, max_y) for the territory's sectors."""
    sectors = territory.ordered_sectors()
    xs = [c.x for c in sectors]
    ys = [c.y for c in sectors]
    return min(xs), min(ys), max(xs), max(ys)


def _d4_transforms(
    cx: float, cy: float,
) -> list[tuple[str, callable]]:
    """Return the eight candidate D_4 transformations centred at (*cx*, *cy*).

    Each entry is (name, function mapping (x, y) -> (x', y')).
    The transforms are: identity, 90-CW, 180, 270-CW, horizontal flip,
    vertical flip, diagonal (transpose), and anti-diagonal.
    """

    def identity(x: int, y: int) -> tuple[float, float]:
        return (float(x), float(y))

    def rot90(x: int, y: int) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return (cx + dy, cy - dx)

    def rot180(x: int, y: int) -> tuple[float, float]:
        return (2 * cx - x, 2 * cy - y)

    def rot270(x: int, y: int) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return (cx - dy, cy + dx)

    def flip_h(x: int, y: int) -> tuple[float, float]:
        return (2 * cx - x, float(y))

    def flip_v(x: int, y: int) -> tuple[float, float]:
        return (float(x), 2 * cy - y)

    def flip_diag(x: int, y: int) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return (cx + dy, cy + dx)

    def flip_anti(x: int, y: int) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return (cx - dy, cy - dx)

    return [
        ("id", identity),
        ("r90", rot90),
        ("r180", rot180),
        ("r270", rot270),
        ("fh", flip_h),
        ("fv", flip_v),
        ("fd", flip_diag),
        ("fa", flip_anti),
    ]


def territory_automorphisms(territory: Territory) -> list[dict[int, int]]:
    """Compute automorphisms of *territory* as sector-index permutations.

    Enumerates D_4 candidate transforms centred on the bounding-box
    centre.  A transform is a valid automorphism iff it maps every
    sector to another sector *and* preserves the 4-adjacency relation.

    Returns a list of dicts mapping ``old_index -> new_index``.
    The identity is always included as the first element.
    """
    sectors = territory.ordered_sectors()
    S = len(sectors)
    if S == 0:
        return [{}]

    coord_set = set(sectors)
    index_by_coord = {c: i for i, c in enumerate(sectors)}

    min_x, min_y, max_x, max_y = _bounding_box(territory)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    # Build physical adjacency set for the territory
    adj_set: set[tuple[int, int]] = set()
    for i, c in enumerate(sectors):
        for nb in neighbors4(c):
            j = index_by_coord.get(nb)
            if j is not None:
                lo, hi = (i, j) if i < j else (j, i)
                adj_set.add((lo, hi))

    automorphisms: list[dict[int, int]] = []
    seen_perms: set[tuple[int, ...]] = set()

    for _name, transform in _d4_transforms(cx, cy):
        perm: dict[int, int] = {}
        valid = True

        for i, c in enumerate(sectors):
            tx, ty = transform(c.x, c.y)
            # Transformed coordinates must be integral
            rx, ry = round(tx), round(ty)
            if abs(tx - rx) > 1e-9 or abs(ty - ry) > 1e-9:
                valid = False
                break
            target = Coord(rx, ry)
            if target not in coord_set:
                valid = False
                break
            perm[i] = index_by_coord[target]

        if not valid:
            continue

        # Verify adjacency preservation
        adj_ok = True
        for i in range(S):
            for j in range(i + 1, S):
                orig_adj = (i, j) in adj_set
                pi, pj = perm[i], perm[j]
                lo, hi = (pi, pj) if pi < pj else (pj, pi)
                mapped_adj = (lo, hi) in adj_set
                if orig_adj != mapped_adj:
                    adj_ok = False
                    break
            if not adj_ok:
                break

        if adj_ok:
            # Deduplicate: multiple transforms can produce the same permutation
            perm_key = tuple(perm[i] for i in range(S))
            if perm_key not in seen_perms:
                seen_perms.add(perm_key)
                automorphisms.append(perm)

    # Identity should always be first; ensure it is
    identity = {i: i for i in range(S)}
    if automorphisms and automorphisms[0] != identity:
        automorphisms = [p for p in automorphisms if p != identity]
        automorphisms.insert(0, identity)

    return automorphisms


def sector_orbits(
    territory: Territory,
    automorphisms: list[dict[int, int]],
) -> list[frozenset[int]]:
    """Partition sector indices into equivalence classes under *automorphisms*.

    Two sector indices are in the same orbit iff some automorphism maps
    one to the other.
    """
    S = len(territory)
    parent = list(range(S))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for perm in automorphisms:
        for i, j in perm.items():
            union(i, j)

    groups: dict[int, set[int]] = {}
    for i in range(S):
        root = find(i)
        groups.setdefault(root, set()).add(i)

    return [frozenset(g) for g in groups.values()]


def orbit_representatives(orbits: list[frozenset[int]]) -> list[int]:
    """Pick the smallest index from each orbit as its canonical representative."""
    return sorted(min(orb) for orb in orbits)


def symmetry_info(territory: Territory) -> SymmetryInfo:
    """Compute and bundle all symmetry data for *territory*."""
    auts = territory_automorphisms(territory)
    orbs = sector_orbits(territory, auts)
    reps = orbit_representatives(orbs)
    return SymmetryInfo(
        automorphisms=tuple(auts),
        orbits=tuple(orbs),
        representatives=tuple(reps),
    )


def canonical_state(
    owner_by_index: tuple[int, ...],
    automorphisms: tuple[dict[int, int], ...],
) -> tuple[tuple[int, ...], dict[int, int]]:
    """Compute the canonical (lex-min) state under the automorphism group.

    For each automorphism σ, compute the transformed ownership vector
    where ``transformed[σ(i)] = owner_by_index[i]``.  Return the lex-min
    vector and the σ that produced it.

    The returned σ maps raw sector indices to canonical indices, so a raw
    action at sector index *c* becomes canonical index ``σ(c)``.
    """
    S = len(owner_by_index)
    best_tuple = owner_by_index
    best_sigma = automorphisms[0]  # identity

    for sigma in automorphisms:
        transformed = [0] * S
        for i in range(S):
            transformed[sigma[i]] = owner_by_index[i]
        t = tuple(transformed)
        if t < best_tuple:
            best_tuple = t
            best_sigma = sigma

    return best_tuple, best_sigma


def invert_automorphism(sigma: dict[int, int]) -> dict[int, int]:
    """Return the inverse permutation of *sigma*.

    Used to map canonical action indices back to raw action indices.
    """
    return {v: k for k, v in sigma.items()}
