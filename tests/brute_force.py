"""Brute-force oracles for tiny RCMAS instances (no SMT).

Everything here follows Chapters 3 and 4 of the thesis directly and uses only
the pure-Python model, so the SMT tests can compare against it:

- :func:`final_states`: every final control vector of a collision-free play;
- :func:`deviations`: every collision-free run in which one agent deviates
  and the others follow a total strategy (Corollary 4.4.2);
- :func:`best_deviation`: ``BR_a(pi)`` of Lemma 4.7.2 by enumeration, with
  the lexicographically smallest optimal action sequence of the deviator;
- :func:`random_play`: a uniformly random play.
"""

from __future__ import annotations

import random
from typing import Callable, Iterator

from rcmas.model import UNOWNED, rewards_from_owner
from rcmas.strategy import CanonicalCompletion, Play, StateKey

JointAction = Callable[[StateKey, int], int]


def final_states(num_sectors: int, num_agents: int, horizon: int) -> Iterator[StateKey]:
    """Every control vector in which each agent controls exactly *horizon* sectors."""
    counts = {a: horizon for a in range(num_agents)}
    counts[UNOWNED] = num_sectors - num_agents * horizon
    if counts[UNOWNED] < 0:
        return
    out: list[int] = []

    def rec() -> Iterator[StateKey]:
        if len(out) == num_sectors:
            yield tuple(out)
            return
        for label in sorted(counts):
            if counts[label] > 0:
                counts[label] -= 1
                out.append(label)
                yield from rec()
                out.pop()
                counts[label] += 1

    yield from rec()


def deviations(
    action: JointAction,
    agent: int,
    num_sectors: int,
    num_agents: int,
    horizon: int,
) -> Iterator[tuple[tuple[int, ...], Play]]:
    """Collision-free runs where *agent* deviates and the others follow *action*.

    Yields ``(sequence of sectors claimed by agent, play)`` in the
    lexicographic order of the sequences.  Runs that collide end in the
    failure state and are not admissible, so they are skipped.
    """

    def rec(
        t: int,
        state: list[int],
        seq: list[int],
        states: list[StateKey],
        profs: list[tuple[int, ...]],
    ):
        if t == horizon:
            yield tuple(seq), Play(tuple(states), tuple(profs), num_agents)
            return
        key = tuple(state)
        opp = {b: action(key, b) for b in range(num_agents) if b != agent}
        used = set(opp.values())
        for i in range(num_sectors):
            if state[i] != UNOWNED or i in used:
                continue
            prof = tuple(i if b == agent else opp[b] for b in range(num_agents))
            nxt = list(state)
            for b, j in enumerate(prof):
                nxt[j] = b
            yield from rec(t + 1, nxt, seq + [i], states + [tuple(nxt)], profs + [prof])

    start = [UNOWNED] * num_sectors
    yield from rec(0, start, [], [tuple(start)], [])


def best_deviation(
    completion: CanonicalCompletion,
    agent: int,
    neighbors: dict,
    demands: tuple[int, ...] | None = None,
    weights: tuple[int, ...] | None = None,
) -> tuple[int, tuple[int, ...], Play]:
    """``BR_a(pi)`` by enumeration (Lemma 4.7.2).

    Returns the best reward over admissible deviations (collision-free and
    every agent meeting its demand), the lexicographically smallest optimal
    sequence of sectors claimed by *agent*, and its play.
    """
    play = completion.play
    n = play.num_agents
    S = len(play.states[0])
    d = demands if demands is not None else (0,) * n
    best: tuple[int, tuple[int, ...], Play] | None = None
    for seq, run in deviations(completion.action, agent, S, n, play.horizon):
        r = rewards_from_owner(run.final_state, n, neighbors, weights)
        if any(r[b] < d[b] for b in range(n)):
            continue
        if best is None or r[agent] > best[0]:
            best = (r[agent], seq, run)
    assert best is not None, "the play itself is an admissible deviation"
    return best


def is_equilibrium(
    play: Play,
    neighbors: dict,
    demands: tuple[int, ...] | None = None,
    weights: tuple[int, ...] | None = None,
) -> bool:
    """True if the canonical completion of *play* is a demand-constrained NE."""
    comp = CanonicalCompletion(play)
    cur = rewards_from_owner(play.final_state, play.num_agents, neighbors, weights)
    return all(
        best_deviation(comp, a, neighbors, demands, weights)[0] <= cur[a]
        for a in range(play.num_agents)
    )


def random_play(
    num_sectors: int, num_agents: int, horizon: int, rng: random.Random
) -> Play:
    """A play whose profiles are uniformly random injective claims."""
    state = [UNOWNED] * num_sectors
    actions = []
    for _ in range(horizon):
        free = [i for i in range(num_sectors) if state[i] == UNOWNED]
        prof = rng.sample(free, num_agents)
        for a, i in enumerate(prof):
            state[i] = a
        actions.append(prof)
    return Play.from_actions(num_sectors, num_agents, actions)
