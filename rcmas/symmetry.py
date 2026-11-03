"""Symmetry detection for Chapter 5 of the thesis.

The tool does not compute the automorphism group ``Aut(T)`` of Definition
5.1.1.  It computes the *geometric subgroup* ``H_geo <= Aut(T)`` of Section
5.1.1: the rotations and reflections of the bounding box of the territory
that map every accessible sector to an accessible sector (Section 7.1).  An
isometry of the grid preserves four-connected adjacency, so every such map is
an automorphism.  ``H_geo`` equals ``Aut(T)`` on the full rectangular grids and
can be trivial when ``Aut(T)`` is not (``5x5-13``, ``6x6-21``, ``6x6-15``).  The
same computation on the bounding box of a subregion or of a connected
component gives the geometric subgroups of Remark 5.1.1 that the local and
component-wise constraints use.

Contents:

- :func:`geometric_automorphisms` (alias :func:`territory_automorphisms`):
  ``H_geo`` as permutations of sector indices;
- :func:`sector_orbits`, :func:`orbit_representatives`, :func:`symmetry_info`:
  the orbits ``T/H`` (Definition 5.1.2) and the representatives ``Reps_H``
  of the spatial canonicalisation constraint (Definition 5.1.7);
- :func:`demand_classes`: the demand classes ``Agt/d`` (Definition 5.1.3),
  with the class of agent 0 (thesis agent 1) first;
- :func:`local_symmetry_info`: for a subregion ``R``, the geometric subgroup
  ``H_R <= Aut(T[R])`` (Definition 5.2.1), the boundary, the problematic
  boundary pairs ``BP(sigma)`` of each element (Definition 5.2.2) and their
  union ``BP(H_R)`` (Corollary 5.2.3);
  :func:`find_maximal_rectangular_subregions` gives the subregions the tool
  uses;
- :func:`component_info`: the connected components, the classes of copies of
  Theorem 5.3.2 with labels that satisfy its hypotheses (i) and (ii), and the
  group ``H`` of that theorem (generators, orbits, ``Reps_H``);
- :class:`SymmetryBreaking` and :func:`symmetry_breaking_plan`: the
  constraints that the tool adds to a query, as data, with a pure-Python test
  :meth:`SymmetryBreaking.admits` of the first-round actions and the final
  state of a play;
- :func:`canonical_state`, :func:`invert_automorphism`: state
  canonicalisation for the Q-learner (not one of the constraints of
  Chapter 5).

Numbering.  Sector indices follow Remark 3.1.1 (bottom row first, tool index
``i`` is thesis sector ``i+1``), and "the smallest sector" of a set is its
smallest index in this enumeration.  Tool agent 0 is thesis agent 1.  The
results of Chapter 5 hold for every enumeration, and the only hypothesis on
the enumeration, (i) and (ii) of Theorem 5.3.2, is checked by
:func:`component_info`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .model import UNOWNED, Coord, Territory

# ---------------------------------------------------------------------------
# Geometric subgroups (Section 5.1.1, Remark 5.1.1, Section 7.1)
# ---------------------------------------------------------------------------

# The eight elements of D_4 as integer matrices (a, b, c, d), acting on an
# offset (dx, dy) from the centre of a bounding box as (a dx + b dy, c dx + d dy).
_D4: tuple[tuple[int, int, int, int], ...] = (
    (1, 0, 0, 1),  # identity
    (0, 1, -1, 0),  # rotation by 90 degrees
    (-1, 0, 0, -1),  # rotation by 180 degrees
    (0, -1, 1, 0),  # rotation by 270 degrees
    (-1, 0, 0, 1),  # left-right reflection
    (1, 0, 0, -1),  # top-bottom reflection
    (0, 1, 1, 0),  # reflection in the diagonal
    (0, -1, -1, 0),  # reflection in the anti-diagonal
)


def _bounding_box_maps(coords: Sequence[Coord]) -> list[dict[Coord, Coord]]:
    """The rotations and reflections of the bounding box of *coords* that map
    the set *coords* onto itself, identity first, without repetitions."""
    if not coords:
        return [{}]
    cells = set(coords)
    # Doubled centre, so that half-integral centres stay integral.
    cx2 = min(c.x for c in coords) + max(c.x for c in coords)
    cy2 = min(c.y for c in coords) + max(c.y for c in coords)
    maps: list[dict[Coord, Coord]] = []
    seen: set[tuple[Coord, ...]] = set()
    for a, b, c, d in _D4:
        image: dict[Coord, Coord] = {}
        for p in coords:
            dx2, dy2 = 2 * p.x - cx2, 2 * p.y - cy2
            x2, y2 = cx2 + a * dx2 + b * dy2, cy2 + c * dx2 + d * dy2
            if x2 % 2 or y2 % 2:
                break
            q = Coord(x2 // 2, y2 // 2)
            if q not in cells:
                break
            image[p] = q
        else:
            key = tuple(image[p] for p in coords)
            if key not in seen:
                seen.add(key)
                maps.append(image)
    return maps


def geometric_subgroup(
    territory: Territory, region: Iterable[int]
) -> list[dict[int, int]]:
    """The geometric subgroup of the induced subgraph ``T[region]``.

    Returns the rotations and reflections of the bounding box of *region* that
    map *region* onto itself, as permutations of the sector indices of
    *region* (identity first).  They preserve four-connected adjacency, so they
    form a subgroup of ``Aut(T[region])`` (Remark 5.1.1).  For a component or
    for the whole territory this is the geometric subgroup ``H_geo`` of that
    component; it is not the full automorphism group.
    """
    sectors = territory.ordered_sectors()
    idx = sorted(set(region))
    coords = [sectors[i] for i in idx]
    out: list[dict[int, int]] = []
    for image in _bounding_box_maps(coords):
        out.append({i: territory.index_of(image[sectors[i]]) for i in idx})  # type: ignore[misc]
    return out


def geometric_automorphisms(territory: Territory) -> list[dict[int, int]]:
    """The geometric subgroup ``H_geo <= Aut(T)`` of the territory (Section 5.1.1).

    Evaluates the eight rotations and reflections of the bounding box of the
    territory and keeps those that map every accessible sector to an
    accessible sector (Section 7.1).  Returns permutations ``i -> sigma(i)`` of
    all sector indices, identity first.  ``Aut(T)`` itself is not computed.
    """
    return geometric_subgroup(territory, range(len(territory)))


def territory_automorphisms(territory: Territory) -> list[dict[int, int]]:
    """Alias of :func:`geometric_automorphisms`: returns ``H_geo``, not ``Aut(T)``."""
    return geometric_automorphisms(territory)


def _orbits(
    domain: Iterable[int], perms: Iterable[dict[int, int]]
) -> list[frozenset[int]]:
    """Orbits of *domain* under the group generated by *perms* (union-find),
    sorted by their smallest element."""
    parent = {i: i for i in domain}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for perm in perms:
        for i, j in perm.items():
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, set[int]] = {}
    for i in parent:
        groups.setdefault(find(i), set()).add(i)
    return sorted((frozenset(g) for g in groups.values()), key=min)


def sector_orbits(
    territory: Territory,
    automorphisms: Iterable[dict[int, int]],
) -> list[frozenset[int]]:
    """The orbits ``T/H`` of the sectors under the group generated by
    *automorphisms* (Definition 5.1.2), sorted by their smallest sector."""
    return _orbits(range(len(territory)), automorphisms)


def orbit_representatives(orbits: Iterable[frozenset[int]]) -> list[int]:
    """``Reps_H``: the smallest sector of each orbit (Definition 5.1.7), sorted."""
    return sorted(min(orb) for orb in orbits)


@dataclass(frozen=True, slots=True)
class SymmetryInfo:
    """A subgroup ``H <= Aut(T)`` with its orbits and representatives.

    Attributes:
        automorphisms: the elements of ``H`` as permutations of sector
            indices, identity first.  :func:`symmetry_info` gives
            ``H = H_geo``.
        orbits: the orbits ``T/H``, sorted by their smallest sector.
        representatives: ``Reps_H``, the smallest sector of each orbit.
    """

    automorphisms: tuple[dict[int, int], ...]
    orbits: tuple[frozenset[int], ...]
    representatives: tuple[int, ...]


def symmetry_info(territory: Territory) -> SymmetryInfo:
    """``H_geo`` of the territory with its orbits and representatives."""
    auts = geometric_automorphisms(territory)
    orbs = sector_orbits(territory, auts)
    return SymmetryInfo(
        automorphisms=tuple(auts),
        orbits=tuple(orbs),
        representatives=tuple(orbit_representatives(orbs)),
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
    action at sector index *c* becomes canonical index ``σ(c)``.  This serves
    the state representation of the Q-learner; it is not one of the
    constraints of Chapter 5.
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


def demand_classes(demands: Sequence[int]) -> list[list[int]]:
    """The demand classes ``Agt/d`` (Definition 5.1.3).

    Agents with the same demand form one class.  Each class lists its agents
    in increasing order, and the classes are ordered by their first agent, so
    the class of agent 0 (thesis agent 1) comes first, as the key of Lemma
    5.1.2 requires.

    >>> demand_classes((2, 2, 5, 5))
    [[0, 1], [2, 3]]
    """
    groups: dict[int, list[int]] = {}
    for agent_idx, d in enumerate(demands):
        groups.setdefault(d, []).append(agent_idx)
    return [sorted(g) for g in groups.values()]


def invert_automorphism(sigma: dict[int, int]) -> dict[int, int]:
    """Return the inverse permutation of *sigma*."""
    return {v: k for k, v in sigma.items()}


# ---------------------------------------------------------------------------
# Local symmetry (Section 5.2): subregions, boundary pairs, BP(H_R)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalSymmetryInfo:
    """Local symmetry data of a subregion ``R`` (Section 5.2).

    Attributes:
        subregion: the sector indices of ``R``.
        automorphisms: the geometric subgroup ``H_R <= Aut(T[R])`` of ``R``
            (rotations and reflections of the bounding box of ``R`` that map
            ``R`` onto itself), as permutations of the indices of ``R``,
            identity first.
        orbits: the orbits ``R/H_R``.
        representatives: ``Reps_R``, the smallest sector of each orbit.
        boundary: ``dR``, the sectors of ``R`` with a neighbour outside ``R``
            (Definition 5.2.1).
        boundary_pairs: ``BP(sigma)`` (Definition 5.2.2) for each
            non-identity element ``sigma`` with ``BP(sigma)`` non-empty,
            keyed by the extension of ``sigma`` to ``T`` as a tuple.
    """

    subregion: frozenset[int]
    automorphisms: tuple[dict[int, int], ...]
    orbits: tuple[frozenset[int], ...]
    representatives: tuple[int, ...]
    boundary: frozenset[int]
    boundary_pairs: dict[tuple[int, ...], frozenset[tuple[int, int]]]

    @property
    def all_boundary_pairs(self) -> frozenset[tuple[int, int]]:
        """``BP(H_R)``, the union of ``BP(sigma)`` over ``sigma in H_R``
        (the antecedent of the constraint ``L_l`` of Corollary 5.2.3)."""
        out: set[tuple[int, int]] = set()
        for bp in self.boundary_pairs.values():
            out.update(bp)
        return frozenset(out)


def _subregion_boundary(
    territory: Territory, subregion: frozenset[int]
) -> frozenset[int]:
    """``dR``: the sectors of *subregion* with a neighbour outside it."""
    nbrs = territory.neighbor_map()
    return frozenset(i for i in subregion if any(j not in subregion for j in nbrs[i]))


def _subregion_automorphisms(
    territory: Territory, subregion: frozenset[int]
) -> list[dict[int, int]]:
    """The geometric subgroup of ``T[subregion]`` (see :func:`geometric_subgroup`)."""
    return geometric_subgroup(territory, subregion)


def _problematic_boundary_pairs(
    territory: Territory,
    subregion: frozenset[int],
    boundary: frozenset[int],
    sigma_r: dict[int, int],
) -> frozenset[tuple[int, int]]:
    """``BP(sigma_R)`` (Definition 5.2.2): the pairs ``(i, j)`` with
    ``i in dR``, ``j`` outside ``R``, ``i ~ j`` and ``sigma_R(i) !~ j``."""
    nbrs = territory.neighbor_map()
    return frozenset(
        (i, j)
        for i in boundary
        for j in nbrs[i]
        if j not in subregion and j not in nbrs[sigma_r[i]]
    )


def local_symmetry_info(
    territory: Territory, subregion: frozenset[int]
) -> LocalSymmetryInfo:
    """Local symmetry data of *subregion* under its geometric subgroup."""
    if not subregion:
        raise ValueError("subregion must be non-empty")
    if not subregion.issubset(set(range(len(territory)))):
        raise ValueError("subregion indices must be valid sector indices")

    auts = _subregion_automorphisms(territory, subregion)
    orbits = _orbits(sorted(subregion), auts)
    boundary = _subregion_boundary(territory, subregion)
    S = len(territory)
    bp_map: dict[tuple[int, ...], frozenset[tuple[int, int]]] = {}
    for aut in auts[1:]:
        bp = _problematic_boundary_pairs(territory, subregion, boundary, aut)
        if bp:
            bp_map[tuple(aut.get(i, i) for i in range(S))] = bp
    return LocalSymmetryInfo(
        subregion=subregion,
        automorphisms=tuple(auts),
        orbits=tuple(orbits),
        representatives=tuple(orbit_representatives(orbits)),
        boundary=boundary,
        boundary_pairs=bp_map,
    )


def find_maximal_rectangular_subregions(territory: Territory) -> list[frozenset[int]]:
    """The subregions ``R_l`` for which the tool builds local constraints.

    These are the maximal axis-parallel rectangles of accessible sectors with
    at least four sectors, other than the whole territory, whose geometric
    subgroup is non-trivial.  They may overlap (Corollary 5.2.3 allows it).
    The result is sorted by decreasing size, then by the sorted sector
    indices.
    """
    sectors = territory.ordered_sectors()
    S = len(sectors)
    if S == 0:
        return []
    xs = sorted({c.x for c in sectors})
    ys = sorted({c.y for c in sectors})
    found: list[frozenset[int]] = []
    for x0 in xs:
        for x1 in (x for x in xs if x >= x0):
            for y0 in ys:
                for y1 in (y for y in ys if y >= y0):
                    if (x1 - x0 + 1) * (y1 - y0 + 1) < 4:
                        continue
                    rect: set[int] = set()
                    for x in range(x0, x1 + 1):
                        for y in range(y0, y1 + 1):
                            i = territory.index_of(Coord(x, y))
                            if i is None:
                                break
                            rect.add(i)
                        else:
                            continue
                        break
                    else:
                        region = frozenset(rect)
                        group = _subregion_automorphisms(territory, region)
                        if len(region) < S and len(group) > 1:
                            found.append(region)
    found.sort(key=lambda r: (-len(r), sorted(r)))
    maximal: list[frozenset[int]] = []
    for cand in found:
        if not any(cand < big for big in maximal):
            maximal.append(cand)
    return maximal


# ---------------------------------------------------------------------------
# Disconnected territories (Section 5.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """Symmetry data of a (possibly disconnected) territory (Section 5.3).

    Attributes:
        components: the connected components ``K_1, ..., K_r`` as sets of
            sector indices, ordered by their smallest sector.
        iso_classes: the components grouped by shape (indices into
            *components*): two components are in one group if one of the eight
            rotations and reflections followed by a translation maps one onto
            the other (Section 7.1).
        component_automorphisms: the geometric subgroup of each component.
        component_orbits: the orbits of each component under its geometric
            subgroup.
        component_representatives: their smallest sectors.
        isomorphisms: for components ``(i, j)`` of one shape group, a
            congruence from ``K_i`` onto ``K_j`` as a sector-index map.
        copy_classes: the classes ``X_1, ..., X_k`` of copies to which the
            tool applies Theorem 5.3.2, each a tuple of component indices
            ordered by smallest sector.  Within a class the order-preserving
            bijection ``K_l -> K_l'`` is an isomorphism and maps every sector
            to a larger one, so the labels below satisfy hypotheses (i) and
            (ii) of the theorem.  A shape group whose copies do not meet these
            hypotheses (for instance a rectangle and its rotation by 90
            degrees) is split into several classes.
        labels: the label ``lambda_l(y)`` of every sector ``y``: its rank
            (from 0) among the sectors of its component.
        generators: generators of the group ``H`` of Theorem 5.3.2, with
            ``H_j`` the geometric subgroup of the first copy of each class:
            the elements of ``H_j`` on that copy, and the exchanges of
            consecutive copies by labels.  Each generator is a permutation of
            all sector indices.
        orbits: the orbits ``T/H``, sorted by their smallest sector.
        representatives: ``Reps_H`` (constraint (2) of Theorem 5.3.2).
    """

    components: tuple[frozenset[int], ...]
    iso_classes: tuple[tuple[int, ...], ...]
    component_automorphisms: tuple[tuple[dict[int, int], ...], ...]
    component_orbits: tuple[tuple[frozenset[int], ...], ...]
    component_representatives: tuple[tuple[int, ...], ...]
    isomorphisms: dict[tuple[int, int], dict[int, int]]
    copy_classes: tuple[tuple[int, ...], ...]
    labels: tuple[int, ...]
    generators: tuple[dict[int, int], ...]
    orbits: tuple[frozenset[int], ...]
    representatives: tuple[int, ...]

    def consecutive_copies(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        """The pairs ``(K_l, K_l')`` of consecutive copies of each class
        (Section 5.4.4), each copy as its sectors in increasing order, so
        that position ``r`` holds the sector with label ``r``."""
        out = []
        for cls in self.copy_classes:
            for a, b in zip(cls, cls[1:]):
                out.append(
                    (
                        tuple(sorted(self.components[a])),
                        tuple(sorted(self.components[b])),
                    )
                )
        return tuple(out)


def territory_connected_components(territory: Territory) -> list[frozenset[int]]:
    """The connected components of the territory, ordered by smallest sector."""
    nbrs = territory.neighbor_map()
    seen: set[int] = set()
    comps: list[frozenset[int]] = []
    for start in range(len(territory)):
        if start in seen:
            continue
        comp = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for j in nbrs[cur]:
                if j not in comp:
                    comp.add(j)
                    stack.append(j)
        seen |= comp
        comps.append(frozenset(comp))
    return sorted(comps, key=min)


def _are_components_isomorphic(
    territory: Territory, comp_a: frozenset[int], comp_b: frozenset[int]
) -> dict[int, int] | None:
    """A rotation or reflection followed by a translation that maps *comp_a*
    onto *comp_b*, as a sector-index map, or ``None`` (Section 7.1)."""
    if len(comp_a) != len(comp_b):
        return None
    sectors = territory.ordered_sectors()
    pa = [sectors[i] for i in sorted(comp_a)]
    cells_b = {sectors[i] for i in comp_b}
    bx, by = min(c.x for c in cells_b), min(c.y for c in cells_b)
    for a, b, c, d in _D4:
        moved = [(a * p.x + b * p.y, c * p.x + d * p.y) for p in pa]
        dx = bx - min(x for x, _ in moved)
        dy = by - min(y for _, y in moved)
        image = [Coord(x + dx, y + dy) for x, y in moved]
        if set(image) == cells_b:
            return {
                i: territory.index_of(q)  # type: ignore[misc]
                for i, q in zip(sorted(comp_a), image)
            }
    return None


def _order_preserving_copy(
    nbrs: dict[int, frozenset[int]], first: Sequence[int], second: Sequence[int]
) -> bool:
    """True if the order-preserving bijection ``first[r] -> second[r]`` is an
    isomorphism of the induced subgraphs and maps every sector to a larger
    one, which are hypotheses (i) and (ii) of Theorem 5.3.2 for the two copies."""
    if len(first) != len(second):
        return False
    if any(x >= y for x, y in zip(first, second)):
        return False
    pos_a = {s: r for r, s in enumerate(first)}
    pos_b = {s: r for r, s in enumerate(second)}
    for r, s in enumerate(first):
        na = {pos_a[j] for j in nbrs[s] if j in pos_a}
        nb = {pos_b[j] for j in nbrs[second[r]] if j in pos_b}
        if na != nb:
            return False
    return True


def component_info(territory: Territory) -> ComponentInfo:
    """Symmetry data of a territory for Theorem 5.3.2 and Section 5.4.4.

    For a connected territory there is one component, and ``H = H_geo``.
    """
    comps = territory_connected_components(territory)
    nbrs = territory.neighbor_map()
    S = len(territory)

    # Shape groups (congruence by D_4 and a translation).
    iso_witnesses: dict[tuple[int, int], dict[int, int]] = {}
    class_of: dict[int, int] = {}
    shape_groups: list[list[int]] = []
    for i in range(len(comps)):
        if i in class_of:
            continue
        class_of[i] = len(shape_groups)
        group = [i]
        for j in range(i + 1, len(comps)):
            if j in class_of:
                continue
            w = _are_components_isomorphic(territory, comps[i], comps[j])
            if w is not None:
                class_of[j] = class_of[i]
                group.append(j)
                iso_witnesses[(i, j)] = w
                iso_witnesses[(j, i)] = invert_automorphism(w)
        shape_groups.append(group)

    # Classes of copies with order-preserving labels: hypotheses (i), (ii).
    copy_classes: list[list[int]] = []
    for group in shape_groups:
        chains: list[list[int]] = []
        for ci in group:
            sec = sorted(comps[ci])
            for chain in chains:
                if _order_preserving_copy(nbrs, sorted(comps[chain[-1]]), sec):
                    chain.append(ci)
                    break
            else:
                chains.append([ci])
        copy_classes.extend(chains)
    copy_classes.sort(key=lambda cls: min(comps[cls[0]]))

    labels = [0] * S
    for comp in comps:
        for r, s in enumerate(sorted(comp)):
            labels[s] = r

    comp_auts = [tuple(geometric_subgroup(territory, comp)) for comp in comps]
    comp_orbits = [
        tuple(_orbits(sorted(comp), auts)) for comp, auts in zip(comps, comp_auts)
    ]
    comp_reps = [tuple(orbit_representatives(orbs)) for orbs in comp_orbits]

    # Generators of H: H_j on the first copy of each class, and exchanges of
    # consecutive copies by labels.
    generators: list[dict[int, int]] = []
    for cls in copy_classes:
        for h in comp_auts[cls[0]][1:]:
            perm = {i: i for i in range(S)}
            perm.update(h)
            generators.append(perm)
        for a, b in zip(cls, cls[1:]):
            perm = {i: i for i in range(S)}
            for x, y in zip(sorted(comps[a]), sorted(comps[b])):
                perm[x], perm[y] = y, x
            generators.append(perm)
    orbits = _orbits(range(S), generators)

    return ComponentInfo(
        components=tuple(comps),
        iso_classes=tuple(tuple(g) for g in shape_groups),
        component_automorphisms=tuple(comp_auts),
        component_orbits=tuple(comp_orbits),
        component_representatives=tuple(comp_reps),
        isomorphisms=iso_witnesses,
        copy_classes=tuple(tuple(c) for c in copy_classes),
        labels=tuple(labels),
        generators=tuple(generators),
        orbits=tuple(orbits),
        representatives=tuple(orbit_representatives(orbits)),
    )


# ---------------------------------------------------------------------------
# The constraints the tool imposes, as data (Section 5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalConstraint:
    """The constraint ``L_l`` of Corollary 5.2.3 for one subregion.

    ``(and over (i, j) in BP(H_R) of not (sec^i_m = sec^j_m and sec^i_m != 0))
    implies (act^1_0 in R implies act^1_0 in Reps_R)``.
    """

    subregion: frozenset[int]
    representatives: frozenset[int]
    boundary_pairs: frozenset[tuple[int, int]]

    @staticmethod
    def of(info: LocalSymmetryInfo) -> LocalConstraint:
        """The constraint ``L_l`` of the subregion described by *info*."""
        return LocalConstraint(
            subregion=info.subregion,
            representatives=frozenset(info.representatives),
            boundary_pairs=info.all_boundary_pairs,
        )

    def holds(self, first_actions: Sequence[int], final_owner: Sequence[int]) -> bool:
        """True if agent 0's first-round action and the final state satisfy ``L_l``."""
        o = first_actions[0]
        if o not in self.subregion or o in self.representatives:
            return True
        return any(
            final_owner[i] == final_owner[j] != UNOWNED for i, j in self.boundary_pairs
        )


@dataclass(frozen=True, slots=True)
class SymmetryBreaking:
    """The symmetry-breaking constraints that the tool adds to a query.

    Attributes:
        num_agents: the number of agents.
        classes: the demand classes (lex-leader constraint, Definition 5.1.6),
            the class of agent 0 first.
        representatives: ``Reps_H`` of the spatial canonicalisation
            constraint (Definition 5.1.7): ``H = H_geo`` on a connected
            territory, the group ``H`` of Theorem 5.3.2 on a disconnected one.
        local: the constraints ``L_l`` of Corollary 5.2.3.
        copy_pairs: consecutive copies ``(K_l, K_l')`` for constraint (3) of
            Theorem 5.3.2, each as its sectors in increasing order (position
            = label).
    """

    num_agents: int
    classes: tuple[tuple[int, ...], ...]
    representatives: frozenset[int]
    local: tuple[LocalConstraint, ...] = ()
    copy_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()

    def lex_leader_holds(self, first_actions: Sequence[int]) -> bool:
        """Definition 5.1.6: first-round actions non-decreasing within each class."""
        return all(
            first_actions[a] <= first_actions[b]
            for cls in self.classes
            for a, b in zip(cls, cls[1:])
        )

    def copy_order_holds(self, first_actions: Sequence[int]) -> bool:
        """Constraint (3) of Theorem 5.3.2 for every pair of consecutive copies.

        With ``O_l`` the sectors of ``K_l`` that agents of agent 0's class
        claim in the first round: if ``O_l'`` is non-empty, then ``O_l`` is
        non-empty and its smallest label is at most the smallest label of
        ``O_l'``.
        """
        if not self.classes:
            return True
        c1 = {first_actions[a] for a in self.classes[0]}
        for earlier, later in self.copy_pairs:
            lab_later = [r for r, s in enumerate(later) if s in c1]
            if not lab_later:
                continue
            lab_earlier = [r for r, s in enumerate(earlier) if s in c1]
            if not lab_earlier or min(lab_earlier) > min(lab_later):
                return False
        return True

    def admits(self, first_actions: Sequence[int], final_owner: Sequence[int]) -> bool:
        """True if a play with these first-round actions (sector of each agent
        in round 0) and this final state satisfies every constraint.

        This is the pure-Python counterpart of
        :func:`rcmas.smt_constraints.symmetry_plan_constraint`.  With no
        rounds (``first_actions`` empty) there is nothing to constrain.
        """
        if len(first_actions) == 0:
            return True
        if not self.lex_leader_holds(first_actions):
            return False
        if first_actions[0] not in self.representatives:
            return False
        if not all(lc.holds(first_actions, final_owner) for lc in self.local):
            return False
        return self.copy_order_holds(first_actions)


def agent_classes(
    num_agents: int, demands: Sequence[int] | None
) -> tuple[tuple[int, ...], ...]:
    """The demand classes of *num_agents* agents; one class without *demands*."""
    if demands is None:
        return (tuple(range(num_agents)),) if num_agents else ()
    if len(demands) != num_agents:
        raise ValueError(f"expected {num_agents} demands, got {len(demands)}")
    return tuple(tuple(c) for c in demand_classes(demands))


def symmetry_breaking_plan(
    territory: Territory,
    num_agents: int,
    demands: Sequence[int] | None = None,
) -> SymmetryBreaking:
    """The constraints that :func:`rcmas.smt_solve.symmetry_constraints` adds.

    - Disconnected territory (Theorem 5.3.2): the lex-leader constraint, the
      spatial canonicalisation constraint under the group ``H`` of the theorem,
      and constraint (3) for the consecutive copies of every class.
    - Connected territory (Theorem 5.1.3): the lex-leader constraint and the
      spatial canonicalisation constraint under ``H_geo``; if ``H_geo`` is
      trivial, also ``L_l`` (Corollary 5.2.3) for every subregion of
      :func:`find_maximal_rectangular_subregions` with ``BP(H_R)`` non-empty.
    """
    classes = agent_classes(num_agents, demands)
    comps = territory_connected_components(territory)
    if len(comps) > 1:
        ci = component_info(territory)
        return SymmetryBreaking(
            num_agents=num_agents,
            classes=classes,
            representatives=frozenset(ci.representatives),
            copy_pairs=ci.consecutive_copies(),
        )
    sym = symmetry_info(territory)
    local: tuple[LocalConstraint, ...] = ()
    if len(sym.automorphisms) <= 1:
        infos = (
            local_symmetry_info(territory, r)
            for r in find_maximal_rectangular_subregions(territory)
        )
        local = tuple(LocalConstraint.of(info) for info in infos if info.boundary_pairs)
    return SymmetryBreaking(
        num_agents=num_agents,
        classes=classes,
        representatives=frozenset(sym.representatives),
        local=local,
    )
