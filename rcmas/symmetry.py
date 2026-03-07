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


def demand_classes(demands: tuple[int, ...]) -> list[list[int]]:
    """Partition agent indices into demand classes.

    Agents with the same demand value are grouped together.  Each inner
    list contains agent indices sorted in ascending order.

    >>> demand_classes((2, 2, 5, 5))
    [[0, 1], [2, 3]]
    """
    groups: dict[int, list[int]] = {}
    for agent_idx, d in enumerate(demands):
        groups.setdefault(d, []).append(agent_idx)
    return [sorted(g) for g in groups.values()]


def invert_automorphism(sigma: dict[int, int]) -> dict[int, int]:
    """Return the inverse permutation of *sigma*.

    Used to map canonical action indices back to raw action indices.
    """
    return {v: k for k, v in sigma.items()}


# ---------------------------------------------------------------------------
# Local symmetry (Ch5 §3): subregion automorphisms, boundary pairs,
# reward-compatible states, conditional symmetry-breaking
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LocalSymmetryInfo:
    """Bundle of local symmetry data for a subregion within a territory.

    Attributes:
        subregion: frozenset of sector indices defining the subregion R
        automorphisms: automorphisms of the induced subgraph T[R]
        orbits: sector-index orbits within R under Aut(T[R])
        representatives: canonical orbit representatives within R
        boundary: sector indices in R that have neighbours outside R
        boundary_pairs: dict mapping each non-identity automorphism (as a
            tuple key) to its set of problematic boundary pairs (i, j)
            where i ∈ ∂R, j ∈ T\\R, i~j, σ_R(i) ≁ j
    """

    subregion: frozenset[int]
    automorphisms: tuple[dict[int, int], ...]
    orbits: tuple[frozenset[int], ...]
    representatives: tuple[int, ...]
    boundary: frozenset[int]
    boundary_pairs: dict[tuple[int, ...], frozenset[tuple[int, int]]]


def _subregion_boundary(
    territory: Territory,
    subregion: frozenset[int],
) -> frozenset[int]:
    """Return sector indices in *subregion* that have neighbours outside it."""
    sectors = territory.ordered_sectors()
    index_by_coord = {c: i for i, c in enumerate(sectors)}
    boundary: set[int] = set()
    for i in subregion:
        for nb in neighbors4(sectors[i]):
            j = index_by_coord.get(nb)
            if j is not None and j not in subregion:
                boundary.add(i)
                break
    return frozenset(boundary)


def _subregion_automorphisms(
    territory: Territory,
    subregion: frozenset[int],
) -> list[dict[int, int]]:
    """Compute automorphisms of the induced subgraph T[R].

    Returns permutations on the *original* sector indices (not re-indexed).
    """
    sectors = territory.ordered_sectors()
    sub_list = sorted(subregion)
    sub_set = set(sub_list)
    index_by_coord = {c: i for i, c in enumerate(sectors)}

    # Build adjacency within R
    adj_set: set[tuple[int, int]] = set()
    for i in sub_list:
        for nb in neighbors4(sectors[i]):
            j = index_by_coord.get(nb)
            if j is not None and j in sub_set:
                lo, hi = (i, j) if i < j else (j, i)
                adj_set.add((lo, hi))

    # Build sub-territory for D4 transform enumeration
    sub_coords = [sectors[i] for i in sub_list]
    coord_to_orig_idx = {sectors[i]: i for i in sub_list}
    sub_coord_set = set(sub_coords)

    xs = [c.x for c in sub_coords]
    ys = [c.y for c in sub_coords]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0

    automorphisms: list[dict[int, int]] = []
    seen_perms: set[tuple[tuple[int, int], ...]] = set()

    for _name, transform in _d4_transforms(cx, cy):
        perm: dict[int, int] = {}
        valid = True
        for i in sub_list:
            c = sectors[i]
            tx, ty = transform(c.x, c.y)
            rx, ry = round(tx), round(ty)
            if abs(tx - rx) > 1e-9 or abs(ty - ry) > 1e-9:
                valid = False
                break
            target = Coord(rx, ry)
            if target not in sub_coord_set:
                valid = False
                break
            perm[i] = coord_to_orig_idx[target]
        if not valid:
            continue

        # Verify adjacency preservation within R
        adj_ok = True
        for i_idx, i in enumerate(sub_list):
            for j in sub_list[i_idx + 1:]:
                orig_adj = (i, j) in adj_set if i < j else (j, i) in adj_set
                pi, pj = perm[i], perm[j]
                lo, hi = (pi, pj) if pi < pj else (pj, pi)
                mapped_adj = (lo, hi) in adj_set
                if orig_adj != mapped_adj:
                    adj_ok = False
                    break
            if not adj_ok:
                break

        if adj_ok:
            perm_key = tuple(sorted(perm.items()))
            if perm_key not in seen_perms:
                seen_perms.add(perm_key)
                automorphisms.append(perm)

    # Ensure identity is first
    identity = {i: i for i in sub_list}
    if automorphisms and automorphisms[0] != identity:
        automorphisms = [p for p in automorphisms if p != identity]
        automorphisms.insert(0, identity)

    return automorphisms


def _problematic_boundary_pairs(
    territory: Territory,
    subregion: frozenset[int],
    boundary: frozenset[int],
    sigma_r: dict[int, int],
) -> frozenset[tuple[int, int]]:
    """Compute BP(σ_R): cross-boundary edges not preserved by σ_R.

    Returns frozenset of (i, j) where i ∈ ∂R, j ∈ T\\R, i~j, σ_R(i) ≁ j.
    """
    sectors = territory.ordered_sectors()
    index_by_coord = {c: i for i, c in enumerate(sectors)}
    pairs: set[tuple[int, int]] = set()

    for i in boundary:
        for nb in neighbors4(sectors[i]):
            j = index_by_coord.get(nb)
            if j is None or j in subregion:
                continue
            # i ∈ ∂R, j ∈ T\R, i ~ j
            sigma_i = sigma_r[i]
            # Check if σ_R(i) ~ j
            sigma_i_coord = sectors[sigma_i]
            j_coord = sectors[j]
            is_adj = abs(sigma_i_coord.x - j_coord.x) + abs(sigma_i_coord.y - j_coord.y) == 1
            if not is_adj:
                pairs.add((i, j))

    return frozenset(pairs)


def local_symmetry_info(
    territory: Territory,
    subregion: frozenset[int],
) -> LocalSymmetryInfo:
    """Compute local symmetry data for a subregion within a territory."""
    if not subregion:
        raise ValueError("subregion must be non-empty")
    if not subregion.issubset(set(range(len(territory)))):
        raise ValueError("subregion indices must be valid sector indices")

    auts = _subregion_automorphisms(territory, subregion)

    # Compute orbits within R using union-find
    sub_list = sorted(subregion)
    parent = {i: i for i in sub_list}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for perm in auts:
        for i, j in perm.items():
            union(i, j)

    groups: dict[int, set[int]] = {}
    for i in sub_list:
        root = find(i)
        groups.setdefault(root, set()).add(i)
    orbits = tuple(frozenset(g) for g in groups.values())
    representatives = tuple(sorted(min(orb) for orb in orbits))

    boundary = _subregion_boundary(territory, subregion)

    # Compute boundary pairs for each non-identity automorphism
    S = len(territory)
    identity = {i: i for i in sub_list}
    bp_map: dict[tuple[int, ...], frozenset[tuple[int, int]]] = {}
    for aut in auts:
        if aut == identity:
            continue
        perm_key = tuple(aut.get(i, i) for i in range(S))
        bp = _problematic_boundary_pairs(territory, subregion, boundary, aut)
        if bp:  # Only store if there are problematic pairs
            bp_map[perm_key] = bp

    return LocalSymmetryInfo(
        subregion=subregion,
        automorphisms=tuple(auts),
        orbits=orbits,
        representatives=representatives,
        boundary=boundary,
        boundary_pairs=bp_map,
    )


def find_maximal_rectangular_subregions(
    territory: Territory,
) -> list[frozenset[int]]:
    """Find maximal rectangular subregions with non-trivial local automorphisms.

    Heuristic: find the largest axis-aligned rectangle(s) that fit within
    the territory and have a non-trivial automorphism group. Returns
    subregions that are not the entire territory (those are handled by
    global symmetry).
    """
    sectors = territory.ordered_sectors()
    S = len(sectors)
    if S == 0:
        return []

    coord_set = set(sectors)
    xs = sorted(set(c.x for c in sectors))
    ys = sorted(set(c.y for c in sectors))

    results: list[frozenset[int]] = []
    index_by_coord = {c: i for i, c in enumerate(sectors)}

    # Try all rectangular regions that contain at least 4 sectors
    for x_start in xs:
        for x_end in xs:
            if x_end < x_start:
                continue
            for y_start in ys:
                for y_end in ys:
                    if y_end < y_start:
                        continue
                    width = x_end - x_start + 1
                    height = y_end - y_start + 1
                    if width * height < 4:
                        continue

                    # Check if all cells in the rectangle are in the territory
                    rect_indices: set[int] = set()
                    complete = True
                    for x in range(x_start, x_end + 1):
                        for y in range(y_start, y_end + 1):
                            idx = index_by_coord.get(Coord(x, y))
                            if idx is None:
                                complete = False
                                break
                            rect_indices.add(idx)
                        if not complete:
                            break

                    if not complete:
                        continue
                    rect = frozenset(rect_indices)

                    # Skip if this is the entire territory
                    if len(rect) == S:
                        continue

                    # Check for non-trivial automorphisms
                    auts = _subregion_automorphisms(territory, rect)
                    if len(auts) > 1:
                        results.append(rect)

    # Remove subsets: keep only maximal rectangles
    maximal: list[frozenset[int]] = []
    results.sort(key=len, reverse=True)
    for rect in results:
        if not any(rect < m for m in maximal):
            maximal.append(rect)

    return maximal


# ---------------------------------------------------------------------------
# Disconnected territory support (Ch5 §4): connected components,
# component isomorphism, wreath product structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """Symmetry data for disconnected territories.

    Attributes:
        components: list of connected components as frozensets of sector indices
        iso_classes: grouping of component indices by isomorphism class.
            Each inner list contains indices into *components* that are
            mutually isomorphic.
        component_automorphisms: per-component automorphism groups
            (permutations on original sector indices)
        component_orbits: per-component orbit partition
        component_representatives: per-component orbit representatives
        isomorphisms: for isomorphic components (i, j), a mapping
            from sector indices in component i to sector indices in
            component j that witnesses the isomorphism
    """

    components: tuple[frozenset[int], ...]
    iso_classes: tuple[tuple[int, ...], ...]
    component_automorphisms: tuple[tuple[dict[int, int], ...], ...]
    component_orbits: tuple[tuple[frozenset[int], ...], ...]
    component_representatives: tuple[tuple[int, ...], ...]
    isomorphisms: dict[tuple[int, int], dict[int, int]]


def territory_connected_components(territory: Territory) -> list[frozenset[int]]:
    """Decompose a territory into connected components (sector-index sets)."""
    sectors = territory.ordered_sectors()
    S = len(sectors)
    if S == 0:
        return []

    index_by_coord = {c: i for i, c in enumerate(sectors)}
    visited: set[int] = set()
    components: list[frozenset[int]] = []

    for start in range(S):
        if start in visited:
            continue
        component: set[int] = set()
        stack = [start]
        visited.add(start)
        while stack:
            cur = stack.pop()
            component.add(cur)
            for nb in neighbors4(sectors[cur]):
                j = index_by_coord.get(nb)
                if j is not None and j not in visited:
                    visited.add(j)
                    stack.append(j)
        components.append(frozenset(component))

    return sorted(components, key=lambda c: min(c))


def _are_components_isomorphic(
    territory: Territory,
    comp_a: frozenset[int],
    comp_b: frozenset[int],
) -> dict[int, int] | None:
    """Test if two components are isomorphic as induced subgraphs.

    Returns a witnessing isomorphism (sector-index mapping A→B), or None.
    Uses the approach: compute automorphisms of comp_a via D4 transforms,
    then try mapping comp_a to comp_b via D4 transforms.
    """
    if len(comp_a) != len(comp_b):
        return None

    sectors = territory.ordered_sectors()
    index_by_coord = {c: i for i, c in enumerate(sectors)}

    # Build adjacency for comp_a and comp_b
    def build_adj(comp: frozenset[int]) -> set[tuple[int, int]]:
        adj: set[tuple[int, int]] = set()
        for i in comp:
            for nb in neighbors4(sectors[i]):
                j = index_by_coord.get(nb)
                if j is not None and j in comp:
                    lo, hi = (i, j) if i < j else (j, i)
                    adj.add((lo, hi))
        return adj

    adj_a = build_adj(comp_a)
    adj_b = build_adj(comp_b)

    # Degree sequences must match
    def degree_seq(comp: frozenset[int], adj: set[tuple[int, int]]) -> tuple[int, ...]:
        deg = {i: 0 for i in comp}
        for lo, hi in adj:
            deg[lo] += 1
            deg[hi] += 1
        return tuple(sorted(deg.values()))

    if degree_seq(comp_a, adj_a) != degree_seq(comp_b, adj_b):
        return None

    # Try all D4 transforms mapping comp_a's bounding box to comp_b's
    coords_a = [sectors[i] for i in sorted(comp_a)]
    coords_b = [sectors[i] for i in sorted(comp_b)]
    coord_b_set = set(coords_b)
    coord_b_to_idx = {sectors[i]: i for i in comp_b}

    # Try translating + D4 transforms
    ax_min, ay_min = min(c.x for c in coords_a), min(c.y for c in coords_a)
    ax_max, ay_max = max(c.x for c in coords_a), max(c.y for c in coords_a)
    bx_min, by_min = min(c.x for c in coords_b), min(c.y for c in coords_b)
    bx_max, by_max = max(c.x for c in coords_b), max(c.y for c in coords_b)

    acx = (ax_min + ax_max) / 2.0
    acy = (ay_min + ay_max) / 2.0

    # For each D4 transform of comp_a, try translating to match comp_b
    for _name, transform in _d4_transforms(acx, acy):
        # Transform all coords in comp_a
        transformed: list[tuple[float, float]] = []
        valid = True
        for c in coords_a:
            tx, ty = transform(c.x, c.y)
            rx, ry = round(tx), round(ty)
            if abs(tx - rx) > 1e-9 or abs(ty - ry) > 1e-9:
                valid = False
                break
            transformed.append((rx, ry))
        if not valid:
            continue

        # Compute translation needed
        t_min_x = min(t[0] for t in transformed)
        t_min_y = min(t[1] for t in transformed)
        dx = bx_min - int(t_min_x)
        dy = by_min - int(t_min_y)

        # Try this translation
        mapping: dict[int, int] = {}
        ok = True
        sorted_a = sorted(comp_a)
        for idx, i in enumerate(sorted_a):
            tx, ty = transformed[idx]
            target = Coord(int(tx) + dx, int(ty) + dy)
            j = coord_b_to_idx.get(target)
            if j is None:
                ok = False
                break
            mapping[i] = j
        if not ok:
            continue

        # Verify adjacency preservation
        adj_ok = True
        sorted_a_list = sorted_a
        for i_pos, i in enumerate(sorted_a_list):
            for j in sorted_a_list[i_pos + 1:]:
                lo, hi = (i, j) if i < j else (j, i)
                orig = (lo, hi) in adj_a
                mi, mj = mapping[i], mapping[j]
                mlo, mhi = (mi, mj) if mi < mj else (mj, mi)
                mapped = (mlo, mhi) in adj_b
                if orig != mapped:
                    adj_ok = False
                    break
            if not adj_ok:
                break
        if adj_ok:
            return mapping

    return None


def component_info(territory: Territory) -> ComponentInfo:
    """Compute symmetry data for a (potentially disconnected) territory."""
    components_list = territory_connected_components(territory)
    n_comp = len(components_list)

    if n_comp <= 1:
        # Connected territory: single component
        comp = components_list[0] if components_list else frozenset()
        auts = _subregion_automorphisms(territory, comp) if comp else [{}]

        sub_list = sorted(comp)
        parent = {i: i for i in sub_list}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for perm in auts:
            for i, j in perm.items():
                union(i, j)

        groups: dict[int, set[int]] = {}
        for i in sub_list:
            root = find(i)
            groups.setdefault(root, set()).add(i)
        orbits = tuple(frozenset(g) for g in groups.values())
        reps = tuple(sorted(min(orb) for orb in orbits))

        return ComponentInfo(
            components=(comp,),
            iso_classes=((0,),),
            component_automorphisms=(tuple(auts),),
            component_orbits=(orbits,),
            component_representatives=(reps,),
            isomorphisms={},
        )

    # Multiple components: test pairwise isomorphism
    iso_map: dict[int, int] = {}  # component_idx -> class_idx
    iso_witnesses: dict[tuple[int, int], dict[int, int]] = {}
    class_counter = 0

    for i in range(n_comp):
        if i in iso_map:
            continue
        iso_map[i] = class_counter
        for j in range(i + 1, n_comp):
            if j in iso_map:
                continue
            witness = _are_components_isomorphic(
                territory, components_list[i], components_list[j]
            )
            if witness is not None:
                iso_map[j] = class_counter
                iso_witnesses[(i, j)] = witness
                # Also store inverse
                iso_witnesses[(j, i)] = {v: k for k, v in witness.items()}
        class_counter += 1

    # Group by iso class
    class_groups: dict[int, list[int]] = {}
    for comp_idx, cls_idx in iso_map.items():
        class_groups.setdefault(cls_idx, []).append(comp_idx)
    iso_classes = tuple(
        tuple(sorted(g)) for g in
        sorted(class_groups.values(), key=lambda g: min(g))
    )

    # Per-component automorphisms and orbits
    comp_auts: list[tuple[dict[int, int], ...]] = []
    comp_orbits: list[tuple[frozenset[int], ...]] = []
    comp_reps: list[tuple[int, ...]] = []

    for comp in components_list:
        auts = _subregion_automorphisms(territory, comp)
        comp_auts.append(tuple(auts))

        sub_list = sorted(comp)
        parent = {i: i for i in sub_list}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for perm in auts:
            for i, j in perm.items():
                union(i, j)

        groups = {}
        for i in sub_list:
            root = find(i)
            groups.setdefault(root, set()).add(i)
        orbits = tuple(frozenset(g) for g in groups.values())
        reps = tuple(sorted(min(orb) for orb in orbits))

        comp_orbits.append(orbits)
        comp_reps.append(reps)

    return ComponentInfo(
        components=tuple(components_list),
        iso_classes=iso_classes,
        component_automorphisms=tuple(comp_auts),
        component_orbits=tuple(comp_orbits),
        component_representatives=tuple(comp_reps),
        isomorphisms=iso_witnesses,
    )
