"""Independent certifier for synthesised equilibria (thesis Section 7.5).

The certifier re-checks a play that a synthesiser reports as a
demand-constrained Nash equilibrium (Definition 3.10.3).  It uses neither the
SMT encoding nor Z3, and it shares no code with the parts of the tool that it
checks: the sector enumeration, the evolution, the rewards, the canonical
completion and the deviation search are all implemented here again, on
bitmasks, from the definitions of Chapters 3 and 4.

:func:`certify` performs these steps.

1. It replays the play from ``s_0`` with the evolution of Definition 3.6.1.
   A profile that claims an occupied sector violates the protocol
   (Definition 3.5.1), and a profile in which two agents claim the same
   sector collides and leads to ``s_bot``; both reject the play.
2. It checks ``phi_sat`` (Definition 3.10.1): the terminal state is a control
   map and ``r_a >= d(a)`` for every agent, with the reward of Definition
   3.9.2 computed by its own connected-component code.
3. It builds the canonical completion of the play (Definition 4.4.2): the
   recorded action at the states ``s_0, ..., s_{m-1}`` of the play, and at
   every other live state the default action, with which tool agent ``b``
   claims the unoccupied sector that has exactly ``b`` unoccupied sectors
   before it in the enumeration of Remark 3.1.1.
4. For every agent ``a`` it computes ``BR_a(pi)`` of Lemma 4.7.2, the largest
   reward that ``a`` obtains by an admissible deviation while the other
   agents follow the completion.  By Corollary 4.4.2 the outcomes of the
   deviations are exactly the collision-free plays in which every other agent
   follows the completion, so a depth-first search over the actions of
   ``a`` covers all of them.  A deviation is admissible if it does not
   collide and every agent meets its demand in the terminal state.  The
   opponents' actions depend only on the state, so the search memoises the
   states it has explored, and it prunes a state when an upper bound on the
   reward that ``a`` can still reach does not exceed the best reward found.
5. It accepts the play if and only if ``BR_a(pi) <= r_a(pi)`` for every
   agent.

The first search starts from the reward ``r_a(pi)``, which the play itself
attains, and stops at the first profitable deviation; for an equilibrium it
is the only search.  Every pruned state has a bound no larger than the best
reward known at the time, so the searches return ``BR_a(pi)`` exactly.  An
agent that already has reward ``m`` needs no search, since every agent
controls exactly ``m`` sectors at the end of a collision-free run (Lemma
3.10.1).

On random satisfying plays of the largest benchmark territories (30 sectors,
2 to 6 agents) a check takes at most a few seconds.

Indices follow the rest of the tool: sectors ``0..|T|-1`` as given by
:meth:`~rcmas.model.Territory.ordered_sectors`, agents ``0..n-1`` (tool agent
``b`` is thesis agent ``b + 1``).  The tool's indices only say which sector an
action claims; the certifier derives the enumeration of Remark 3.1.1 itself
from the coordinates (bottom row first, left to right within a row), so a
fault in the tool's enumeration changes the completion that is checked.

:func:`main` implements ``rcmas certify --results LOG.jsonl``, which
re-checks every equilibrium recorded in a result log of the experiment
runner and prints a summary.  A record counts as an equilibrium if its
``outcome`` is ``"ne"`` (or, without an ``outcome``, if ``found_ne`` is
true); its play is read from ``play`` (or ``play_actions``,
``actions_by_round``, ``actions``), and its grid from ``grid_path``, which may
point to another machine (the part after ``grids/`` is looked up locally).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .model import Coord, Territory

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

REASON_KEYWORDS = (
    "ok",
    "invalid",
    "protocol",
    "collision",
    "unsatisfying",
    "deviation",
    "deadline",
)
"""First word of :attr:`CertResult.reason`, before the colon."""


@dataclass(frozen=True, slots=True)
class CertResult:
    """Outcome of :func:`certify`.

    Attributes:
        accepted: ``True`` if the play is satisfying and its canonical
            completion is a demand-constrained Nash equilibrium.
        reason: ``"ok"``, or why the play was rejected, as
            ``"<keyword>: <detail>"`` with a keyword from
            :data:`REASON_KEYWORDS` (``"deadline"`` means that the check did
            not finish, not that the play is wrong).
        rewards: ``r_a`` on the replayed terminal state (Definition 3.9.2),
            empty if the replay failed.
        best_responses: ``BR_a(pi)`` for every agent against the canonical
            completion (Lemma 4.7.2), empty if the play is not satisfying and
            shorter than ``n`` if the deadline passed during the search.
        states_explored: number of states the deviation searches expanded,
            summed over the agents.
    """

    accepted: bool
    reason: str
    rewards: tuple[int, ...]
    best_responses: tuple[int, ...]
    states_explored: int

    @property
    def keyword(self) -> str:
        """The first word of :attr:`reason`."""
        return self.reason.split(":", 1)[0]


class _Reject(Exception):
    """Internal: the play is rejected for the reason in ``args[0]``."""


class _DeadlinePassed(Exception):
    """Internal: the deadline passed during a search."""


# ---------------------------------------------------------------------------
# Territory as bitmasks (Remark 3.1.1, Section 3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Board:
    """The territory laid out on a padded bit grid.

    Bit ``row * stride + col`` stands for the sector in column ``col`` of row
    ``row``, where row 0 is the bottom row of the bounding box.  The stride is
    one more than the width, so a shift by one never wraps around a row.
    Increasing bit positions therefore list the sectors in the enumeration of
    Remark 3.1.1, and the ``k``-th lowest set bit of a mask is its ``k``-th
    sector in that enumeration.
    """

    num_sectors: int
    stride: int
    valid: int
    bit_of_index: tuple[int, ...]
    bit_of_coord: dict[Coord, int]

    @staticmethod
    def of(territory: Territory) -> _Board:
        coords = list(territory.sectors)
        if not coords:
            return _Board(0, 1, 0, (), {})
        xmin = min(c.x for c in coords)
        xmax = max(c.x for c in coords)
        ymax = max(c.y for c in coords)
        stride = xmax - xmin + 2
        # Coord.y is the line of the grid file, counted from the top, so the
        # bottom row of the territory has the largest y.
        bit_of_coord = {c: 1 << ((ymax - c.y) * stride + (c.x - xmin)) for c in coords}
        valid = 0
        for b in bit_of_coord.values():
            valid |= b
        ordered = territory.ordered_sectors()
        if sorted(ordered, key=lambda c: (c.y, c.x)) != sorted(
            coords, key=lambda c: (c.y, c.x)
        ):
            raise ValueError("territory enumeration does not list its sectors")
        bit_of_index = tuple(bit_of_coord[c] for c in ordered)
        return _Board(len(coords), stride, valid, bit_of_index, bit_of_coord)

    def expand(self, mask: int) -> int:
        """The accessible sectors adjacent to a sector of *mask* (Section 3.2)."""
        w = self.stride
        return ((mask << 1) | (mask >> 1) | (mask << w) | (mask >> w)) & self.valid

    def largest_component(self, mask: int) -> int:
        """Size of the largest four-connected component of *mask* (Definition 3.9.2)."""
        w = self.stride
        best = 0
        while mask:
            comp = mask & -mask
            while True:
                grown = comp | (
                    ((comp << 1) | (comp >> 1) | (comp << w) | (comp >> w)) & mask
                )
                if grown == comp:
                    break
                comp = grown
            size = comp.bit_count()
            if size > best:
                best = size
            mask &= ~comp
        return best


def _lowest_bits(mask: int, count: int) -> list[int]:
    """The *count* lowest set bits of *mask*, as single-bit masks."""
    out: list[int] = []
    for _ in range(count):
        low = mask & -mask
        if not low:
            break
        out.append(low)
        mask ^= low
    return out


# ---------------------------------------------------------------------------
# Upper bound on the reward that the deviator can still reach
# ---------------------------------------------------------------------------


def _reach(board: _Board, piece: int, own: int, free: int, rounds: int) -> int:
    """The sectors within distance *rounds* of *piece*.

    Claiming a free sector costs 1, a sector of *own* costs 0, and every
    other sector is blocked.  *piece* is a connected set of sectors of *own*.
    """
    w = board.stride
    reach = piece
    for _ in range(rounds):
        grown = reach | (
            ((reach << 1) | (reach >> 1) | (reach << w) | (reach >> w)) & free
        )
        if grown == reach:
            break
        reach = grown
        # The sectors of own that the newly reached free sectors join.
        while True:
            grown = reach | (
                ((reach << 1) | (reach >> 1) | (reach << w) | (reach >> w)) & own
            )
            if grown == reach:
                break
            reach = grown
    return reach


def _may_exceed(
    board: _Board, own: int, free: int, rounds: int, threshold: int
) -> bool:
    """False only if the deviator cannot end with a reward above *threshold*.

    *own* is the set of sectors that the deviator controls, *free* the
    unoccupied sectors that it may still claim, and *rounds* the number of
    sectors it will claim.  Its final largest region is connected and
    consists of some sectors of *own* and at most *rounds* claimed sectors.

    - The region lies in one component ``K`` of the subgraph that
      ``own | free`` induces, so it has at most ``|own & K| +
      min(rounds, |free & K|)`` sectors, and at most *rounds* if it contains
      no sector of *own*.
    - If it contains a component ``P`` of *own*, every sector of the region
      is joined to ``P`` by a path inside the region with at most *rounds*
      claimed sectors, so it lies in :func:`_reach` of ``P``.  The count of
      the first item over that set bounds the region again.

    Pieces are checked until one bound exceeds *threshold*.
    """
    if not free:
        return False  # every action collides, so no run is admissible
    if rounds >= threshold + 1 and free.bit_count() >= threshold + 1:
        return True
    w = board.stride
    passable = own | free
    pending = own
    while pending:
        comp = pending & -pending
        while True:
            grown = comp | (
                ((comp << 1) | (comp >> 1) | (comp << w) | (comp >> w)) & passable
            )
            if grown == comp:
                break
            comp = grown
        pending &= ~comp
        mine = comp & own
        if mine.bit_count() + min(rounds, (comp & free).bit_count()) <= threshold:
            continue
        pieces = mine
        while pieces:
            piece = pieces & -pieces
            while True:
                grown = piece | (
                    ((piece << 1) | (piece >> 1) | (piece << w) | (piece >> w)) & own
                )
                if grown == piece:
                    break
                piece = grown
            pieces &= ~piece
            reach = _reach(board, piece, own, free, rounds)
            bound = (reach & own).bit_count() + min(rounds, (reach & free).bit_count())
            if bound > threshold:
                return True
    return False


# ---------------------------------------------------------------------------
# Deviation search (Lemma 4.7.2, Corollary 4.4.2)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Search:
    """Depth-first search over the deviations of one agent ``a``.

    The search looks for an admissible terminal state in which ``a`` has a
    reward above :attr:`best`, raises :attr:`best` to the largest reward it
    finds, and stops as soon as :attr:`best` reaches :attr:`stop`.  When it
    returns without stopping, no admissible deviation gives ``a`` more than
    :attr:`best`.
    """

    board: _Board
    num_agents: int
    horizon: int
    agent: int
    demands: tuple[int, ...]
    play_index: dict[tuple[int, ...], int]
    play_bits: tuple[tuple[int, ...], ...]
    deadline: float | None
    best: int
    stop: int
    explored: int = 0

    def run(self) -> int:
        """Search from ``s_0`` and return :attr:`best`."""
        if self.best < self.stop:
            self._dfs(0, (0,) * self.num_agents, 0, set())
        return self.best

    def _opponents(self, state: tuple[int, ...], occupied: int) -> tuple[int, ...]:
        """The profile of the canonical completion at *state* (Definition 4.4.2)."""
        t = self.play_index.get(state)
        if t is not None:
            return self.play_bits[t]
        return tuple(_lowest_bits(self.board.valid & ~occupied, self.num_agents))

    def _dfs(
        self,
        t: int,
        state: tuple[int, ...],
        occupied: int,
        seen: set[tuple[int, ...]],
    ) -> None:
        # A state is expanded once.  The opponents' actions and the number
        # of rounds left depend only on the state, and when the search
        # returns from a state without stopping, no admissible terminal state
        # below it gives ``a`` more than ``best``, which only grows.
        if state in seen:
            return
        seen.add(state)
        self.explored += 1
        if self.deadline is not None and self.explored & 1023 == 1:
            if time.monotonic() > self.deadline:
                raise _DeadlinePassed
        a = self.agent
        board = self.board
        if t == self.horizon:
            # Terminal state: s_m is not live (Lemma 4.4.1(1)), and the
            # deviation is admissible if every agent meets its demand.
            value = board.largest_component(state[a])
            if value > self.best and all(
                board.largest_component(state[b]) >= self.demands[b]
                for b in range(self.num_agents)
                if self.demands[b] > 0
            ):
                self.best = value
            return
        profile = self._opponents(state, occupied)
        taken = 0
        for b in range(self.num_agents):
            if b != a:
                taken |= profile[b]
        if taken.bit_count() != self.num_agents - 1:
            return  # two opponents collide: every successor is s_bot
        # Claiming a sector that an opponent claims in this round collides.
        options = board.valid & ~occupied & ~taken
        rounds = self.horizon - t
        own = state[a]
        if not _may_exceed(board, own, options, rounds, self.best):
            return
        adjacent = board.expand(own) & options
        for group in (adjacent, options & ~adjacent):
            while group:
                x = 1 << (group.bit_length() - 1)
                group ^= x
                nxt = list(state)
                for b in range(self.num_agents):
                    nxt[b] |= x if b == a else profile[b]
                self._dfs(t + 1, tuple(nxt), occupied | taken | x, seen)
                if self.best >= self.stop:
                    return


def _best_response(make: Any, current: int, horizon: int) -> tuple[int, int]:
    """``BR_a(pi)`` and the number of states explored.

    *make(best, stop)* builds a :class:`_Search`.  The first search proves
    ``BR_a(pi) = r_a(pi)``, which is all an equilibrium needs, or stops at the
    first profitable deviation.  In the second case a search for reward
    ``m`` follows, which most profitable deviations reach and which prunes
    every state whose bound is below ``m``, and if it fails, a
    branch-and-bound search from the reward found gives the exact value.
    """
    if current >= horizon:
        return current, 0  # every agent ends with exactly m sectors
    explored = 0
    search = make(current, current + 1)
    try:
        found = search.run()
        explored += search.explored
        if found == current or found >= horizon:
            return found, explored
        search = make(horizon - 1, horizon)
        value = search.run()
        explored += search.explored
        if value >= horizon:
            return value, explored
        search = make(found, horizon - 1)
        value = search.run()
        explored += search.explored
    except _DeadlinePassed:
        raise _DeadlinePassed(explored + search.explored) from None
    return value, explored


# ---------------------------------------------------------------------------
# certify
# ---------------------------------------------------------------------------


def _action_bit(board: _Board, action: Any) -> int:
    """The bit of the sector that *action* claims.

    An action is a tool sector index, a :class:`~rcmas.model.Coord`, a pair
    ``(x, y)`` of grid-file coordinates, or a mapping with keys ``x`` and ``y``.
    """
    if isinstance(action, bool):
        raise _Reject(f"invalid: action {action!r} is not a sector")
    if isinstance(action, int):
        if 0 <= action < board.num_sectors:
            return board.bit_of_index[action]
        raise _Reject(f"invalid: sector index {action} is out of range")
    if isinstance(action, Coord):
        coord = action
    elif isinstance(action, dict) and "x" in action and "y" in action:
        coord = Coord(int(action["x"]), int(action["y"]))
    elif (
        isinstance(action, (list, tuple))
        and len(action) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in action)
    ):
        coord = Coord(action[0], action[1])
    else:
        raise _Reject(f"invalid: action {action!r} is not a sector")
    bit = board.bit_of_coord.get(coord)
    if bit is None:
        raise _Reject(f"invalid: sector {coord} is not in the territory")
    return bit


def _replay(
    board: _Board, n: int, actions_by_round: Sequence[Sequence[Any]]
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], tuple[int, ...]]:
    """Replay the profiles from ``s_0`` with the evolution of Definition 3.6.1.

    Returns the states ``s_0, ..., s_{m-1}`` (as per-agent masks), the
    profiles (as per-agent bits) and the terminal state.
    """
    masks = [0] * n
    occupied = 0
    states: list[tuple[int, ...]] = []
    profiles: list[tuple[int, ...]] = []
    for t, raw in enumerate(actions_by_round):
        prof = list(raw)
        if len(prof) != n:
            raise _Reject(f"invalid: round {t} has {len(prof)} actions for {n} agents")
        if (board.valid & ~occupied).bit_count() < n:
            raise _Reject(f"invalid: state s_{t} is not live, the run has ended")
        bits = [_action_bit(board, act) for act in prof]
        for a, bit in enumerate(bits):
            if bit & occupied:
                raise _Reject(
                    f"protocol: in round {t} agent {a} claims the occupied "
                    f"sector {_index_of_bit(board, bit)}"
                )
        for a in range(n):
            for b in range(a + 1, n):
                if bits[a] == bits[b]:
                    raise _Reject(
                        f"collision: in round {t} agents {a} and {b} both claim "
                        f"sector {_index_of_bit(board, bits[a])}"
                    )
        states.append(tuple(masks))
        profiles.append(tuple(bits))
        for a, bit in enumerate(bits):
            masks[a] |= bit
            occupied |= bit
    return states, profiles, tuple(masks)


def _index_of_bit(board: _Board, bit: int) -> int:
    return board.bit_of_index.index(bit)


def certify(
    territory: Territory,
    n_agents: int,
    horizon: int | None,
    demands: Sequence[int] | None,
    actions_by_round: Iterable[Sequence[Any]],
    *,
    deadline: float | None = None,
) -> CertResult:
    """Check that the canonical completion of a play is a demand-constrained NE.

    *actions_by_round* lists the profiles ``ap_0, ..., ap_{m-1}`` of the
    play; ``actions_by_round[t][a]`` is the sector that agent ``a`` claims in
    round ``t``, as a tool sector index or a coordinate (see the module
    docstring).  *horizon* must be ``m = floor(|T|/n)``; ``None`` selects it.
    *demands* defaults to 0 for every agent.  *deadline* is a value of
    :func:`time.monotonic` after which the search stops with reason
    ``"deadline"``; ``None`` means no limit.

    Returns a :class:`CertResult`.  The play is accepted if it replays without
    a protocol violation or collision, it satisfies ``phi_sat``, and no agent
    has an admissible deviation that raises its reward against the canonical
    completion of the play.
    """
    explored = 0
    rewards: tuple[int, ...] = ()
    responses: list[int] = []
    try:
        if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents < 1:
            raise _Reject(f"invalid: number of agents {n_agents!r}")
        board = _Board.of(territory)
        n = n_agents
        m = board.num_sectors // n
        if horizon is not None and horizon != m:
            raise _Reject(f"invalid: horizon {horizon} is not floor(|T|/n) = {m}")
        d = tuple(0 for _ in range(n)) if demands is None else tuple(demands)
        if len(d) != n or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in d
        ):
            raise _Reject(f"invalid: demands {demands!r} for {n} agents")
        try:
            rounds = [list(p) for p in actions_by_round]
        except TypeError:
            raise _Reject("invalid: the play is not a list of profiles") from None
        if len(rounds) != m:
            raise _Reject(f"invalid: the play has {len(rounds)} rounds, m = {m}")

        states, profiles, final = _replay(board, n, rounds)
        rewards = tuple(board.largest_component(final[a]) for a in range(n))
        short = [a for a in range(n) if rewards[a] < d[a]]
        if short:
            a = short[0]
            raise _Reject(
                f"unsatisfying: agent {a} has reward {rewards[a]} below its "
                f"demand {d[a]}"
            )

        play_index = {s: t for t, s in enumerate(states)}
        if len(play_index) != len(states):  # pragma: no cover - Lemma 4.4.1(3)
            raise _Reject("invalid: the play repeats a state")
        for a in range(n):

            def make(best: int, stop: int, a: int = a) -> _Search:
                return _Search(
                    board=board,
                    num_agents=n,
                    horizon=m,
                    agent=a,
                    demands=d,
                    play_index=play_index,
                    play_bits=tuple(profiles),
                    deadline=deadline,
                    best=best,
                    stop=stop,
                )

            value, count = _best_response(make, rewards[a], m)
            explored += count
            responses.append(value)

        better = [a for a in range(n) if responses[a] > rewards[a]]
        if better:
            detail = ", ".join(
                f"agent {a} reaches {responses[a]} > {rewards[a]}" for a in better
            )
            raise _Reject(f"deviation: {detail}")
    except _Reject as rej:
        return CertResult(False, str(rej.args[0]), rewards, tuple(responses), explored)
    except _DeadlinePassed as late:
        return CertResult(
            False,
            f"deadline: the search for agent {len(responses)} did not finish",
            rewards,
            tuple(responses),
            explored + (late.args[0] if late.args else 0),
        )
    return CertResult(True, "ok", rewards, tuple(responses), explored)


# ---------------------------------------------------------------------------
# Result logs (rcmas certify --results)
# ---------------------------------------------------------------------------

_NO_PLAY = "the record has no play"
_PLAY_KEYS = ("play", "play_actions", "actions_by_round", "actions")
_AGENT_KEYS = ("num_agents", "n_agents", "agents")
_REWARD_KEYS = ("rewards", "payoff", "payoff_by_agent")


def claims_equilibrium(record: dict[str, Any]) -> bool:
    """True if a result record reports an equilibrium.

    Uses the ``outcome`` field when it is present, otherwise ``found_ne``.
    """
    if "outcome" in record and record["outcome"] is not None:
        return record["outcome"] == "ne"
    return bool(record.get("found_ne"))


def record_play(record: dict[str, Any]) -> list[list[Any]] | None:
    """The recorded profiles of the play, or ``None`` if the record has none."""
    for key in _PLAY_KEYS:
        value = record.get(key)
        if isinstance(value, dict):
            value = value.get("actions", value.get("actions_by_round"))
        if isinstance(value, list):
            return value
    return None


def resolve_grid(
    record: dict[str, Any], results_path: Path, grid_root: Path | None = None
) -> Path | None:
    """Find the grid file of a record.

    Tries the recorded path, then the part after the last ``grids/``
    component below *grid_root*, below the ``grids`` directory next to the
    results file's parent, and below the tool's own ``grids`` directory, so
    a log written on another machine can be checked.
    """
    raw = record.get("grid_path") or record.get("grid")
    if not raw:
        return None
    path = Path(raw)
    candidates = [path, results_path.parent / path]
    text = str(raw).replace("\\", "/")
    if "grids/" in text:
        tail = text.rsplit("grids/", 1)[1]
    else:
        tail = text
    roots = [grid_root] if grid_root is not None else []
    roots += [
        results_path.parent / "grids",
        results_path.parent.parent / "grids",
        Path(__file__).resolve().parent.parent / "grids",
    ]
    candidates += [root / tail for root in roots]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _record_name(record: dict[str, Any], lineno: int) -> str:
    name = record.get("name")
    if name:
        run = record.get("run")
        return f"{name}#{run}" if run is not None else str(name)
    return f"line {lineno}"


def _config_of(record: dict[str, Any]) -> str:
    algo = str(record.get("algorithm", "?"))
    return algo + ("+sym" if record.get("symmetry") else "")


def _load_territory(path: Path, cache: dict[Path, Territory]) -> Territory:
    if path not in cache:
        with open(path, encoding="utf-8") as f:
            cache[path] = Territory.from_ascii(f)
    return cache[path]


def certify_record(
    record: dict[str, Any],
    results_path: Path,
    *,
    grid_root: Path | None = None,
    timeout_s: float | None = None,
    cache: dict[Path, Territory] | None = None,
) -> tuple[str, CertResult | None, str]:
    """Certify one result record.

    Returns ``(status, result, message)`` with status ``"accepted"``,
    ``"rejected"`` (the certifier refuted the play, or its rewards differ
    from the recorded ones) or ``"unchecked"`` (no play, no grid, or the
    deadline passed).
    """
    cache = {} if cache is None else cache
    play = record_play(record)
    if play is None:
        return "unchecked", None, _NO_PLAY
    grid = resolve_grid(record, results_path, grid_root)
    if grid is None:
        return "unchecked", None, f"grid not found: {record.get('grid_path')!r}"
    n = next((record[k] for k in _AGENT_KEYS if record.get(k) is not None), None)
    if not isinstance(n, int):
        return "unchecked", None, "the record has no number of agents"
    territory = _load_territory(grid, cache)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    res = certify(
        territory,
        n,
        record.get("horizon"),
        record.get("demands"),
        play,
        deadline=deadline,
    )
    if res.keyword == "deadline":
        return "unchecked", res, res.reason
    if not res.accepted:
        return "rejected", res, res.reason
    recorded = next((record[k] for k in _REWARD_KEYS if record.get(k)), None)
    if recorded is not None and tuple(recorded) != res.rewards:
        return (
            "rejected",
            res,
            f"rewards: recorded {tuple(recorded)}, replayed {res.rewards}",
        )
    return "accepted", res, "ok"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of ``rcmas certify`` to *parser*."""
    parser.add_argument(
        "--results",
        required=True,
        nargs="+",
        type=Path,
        help="Result log(s) of the experiment runner (JSON lines)",
    )
    parser.add_argument(
        "--grid-root",
        type=Path,
        default=None,
        help="Directory that holds the grid files (default: next to the log, then the tool's grids/)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Time limit in seconds for each check (default: none)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write one JSON line per checked equilibrium to this file",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print a line for every equilibrium"
    )


def main(argv: list[str] | None = None) -> int:
    """``rcmas certify``: re-check every equilibrium of one or more result logs.

    Exit status: 0 if every recorded equilibrium was accepted, 1 if any was
    rejected, 2 if none was rejected but some could not be checked.
    """
    parser = argparse.ArgumentParser(
        prog="rcmas certify",
        description=(
            "Re-check every equilibrium in a result log with the independent "
            "certifier of thesis Section 7.5 (no SMT, no Z3)."
        ),
    )
    add_arguments(parser)
    ns = parser.parse_args(argv)
    return run_logs(
        ns.results,
        grid_root=ns.grid_root,
        timeout_s=ns.timeout,
        output=ns.output,
        verbose=ns.verbose,
    )


def run_logs(
    paths: Sequence[Path],
    *,
    grid_root: Path | None = None,
    timeout_s: float | None = None,
    output: Path | None = None,
    verbose: bool = False,
    stream: Any = None,
) -> int:
    """Certify every equilibrium in the logs at *paths* and print a summary."""
    out = sys.stdout if stream is None else stream
    cache: dict[Path, Territory] = {}
    counts = {"accepted": 0, "rejected": 0, "unchecked": 0}
    by_config: dict[str, dict[str, int]] = {}
    records = 0
    no_play = 0
    slowest = (0.0, "")
    start = time.monotonic()
    sink = open(output, "w", encoding="utf-8") if output is not None else None
    try:
        for path in paths:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            for lineno, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as err:
                    print(f"{path}:{lineno}: unreadable record ({err})", file=out)
                    counts["unchecked"] += 1
                    continue
                if not isinstance(record, dict) or not claims_equilibrium(record):
                    continue
                name = _record_name(record, lineno)
                t0 = time.monotonic()
                status, res, message = certify_record(
                    record,
                    path,
                    grid_root=grid_root,
                    timeout_s=timeout_s,
                    cache=cache,
                )
                elapsed = time.monotonic() - t0
                if res is not None and elapsed > slowest[0]:
                    slowest = (elapsed, name)
                counts[status] += 1
                cfg = by_config.setdefault(
                    _config_of(record), {"accepted": 0, "rejected": 0, "unchecked": 0}
                )
                cfg[status] += 1
                quiet = res is None and message == _NO_PLAY
                no_play += quiet
                if verbose or not (status == "accepted" or quiet):
                    print(f"{status:9s} {name}: {message} ({elapsed:.2f} s)", file=out)
                if sink is not None:
                    sink.write(
                        json.dumps(
                            {
                                "name": record.get("name"),
                                "run": record.get("run"),
                                "results": str(path),
                                "line": lineno,
                                "status": status,
                                "message": message,
                                "rewards": list(res.rewards) if res else None,
                                "best_responses": (
                                    list(res.best_responses) if res else None
                                ),
                                "states_explored": res.states_explored if res else 0,
                                "time_s": round(elapsed, 3),
                            }
                        )
                        + "\n"
                    )
    finally:
        if sink is not None:
            sink.close()
    total = sum(counts.values())
    print(
        f"certify: {records} records, {total} equilibria: "
        f"{counts['accepted']} accepted, {counts['rejected']} rejected, "
        f"{counts['unchecked']} unchecked ({time.monotonic() - start:.1f} s)",
        file=out,
    )
    for cfg_name in sorted(by_config):
        c = by_config[cfg_name]
        print(
            f"  {cfg_name:20s} accepted {c['accepted']}, rejected {c['rejected']}, "
            f"unchecked {c['unchecked']}",
            file=out,
        )
    if no_play:
        print(f"  {no_play} equilibria have no recorded play", file=out)
    if slowest[1]:
        print(f"  slowest check: {slowest[1]} ({slowest[0]:.2f} s)", file=out)
    if counts["rejected"]:
        return 1
    if counts["unchecked"]:
        return 2
    return 0
