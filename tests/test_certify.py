"""Tests for the independent certifier (thesis Section 7.5, ``rcmas.certify``).

The certifier is compared with two oracles that it shares no code with:

- a literal enumeration of the deviation *strategies* of an agent
  (Definition 3.8.1) on the tiniest instances, with the outcome of
  Definition 3.9.1 and the reward of Definition 3.9.2;
- the enumeration of deviations as action sequences in ``tests/brute_force``
  (Corollary 4.4.2), which uses ``rcmas.strategy.CanonicalCompletion`` and
  ``rcmas.model.rewards_from_owner``.

It must also reject a play that the old payoff-floor check of IBIS accepted,
reject colliding and unsatisfying plays, and accept the equilibria that
IBIS, S-IBIS, Q-IBIS and CEGAR-NE return.
"""

from __future__ import annotations

import functools
import io
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rcmas import certify as cert_mod
from rcmas.certify import CertResult, certify, run_logs
from rcmas.model import UNOWNED, Coord, Territory, rewards_from_owner
from rcmas.strategy import CanonicalCompletion, Play

from .brute_force import best_deviation, random_play

GRIDS = Path(__file__).resolve().parent.parent / "grids"


def _t(*rows: str) -> Territory:
    return Territory.from_ascii(list(rows))


def _running_example() -> Territory:
    with open(GRIDS / "asymmetric" / "interesting.txt", encoding="utf-8") as f:
        return Territory.from_ascii(f)


def _thesis_play(columns: list[list[int]]) -> list[list[int]]:
    """Profiles from per-agent claim sequences in thesis numbering (1-based)."""
    rounds = len(columns[0])
    return [[col[t] - 1 for col in columns] for t in range(rounds)]


# Example 4.4.1: the play of Figure 3.1 (rewards (2, 4, 4), Example 3.9.1).
FIG_3_1 = _thesis_play([[11, 10, 5, 1], [8, 7, 12, 9], [3, 2, 6, 4]])

# An S-IBIS output of the tool before the revision on the running example
# (working note 01, defect M1): a 7,9,6,8 / b 5,11,12,10 / c 1,4,3,2.
OLD_SIBIS_PLAY = _thesis_play([[7, 9, 6, 8], [5, 11, 12, 10], [1, 4, 3, 2]])


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


def _strategy_enumeration_br(
    territory: Territory, play: Play, agent: int, demands: tuple[int, ...]
) -> int:
    """``BR_a(pi)`` by enumerating the strategies of *agent* (Definition 3.8.1).

    A strategy assigns an available action to every live state.  Only the
    states that some deviation reaches can influence the outcome, so the
    oracle enumerates every assignment of available sectors to those states,
    runs the outcome of Definition 3.9.1 (a collision ends in ``s_bot``,
    which is not satisfying) and keeps the best reward of a satisfying
    outcome.  The other agents follow the canonical completion.
    """
    n = play.num_agents
    S = len(territory)
    comp = CanonicalCompletion(play)
    nb = territory.neighbor_map()

    def live(s: tuple[int, ...]) -> bool:
        return sum(1 for o in s if o == UNOWNED) >= n

    # Live states that some choice of the deviator reaches.
    reach: list[tuple[int, ...]] = []
    frontier = [tuple([UNOWNED] * S)]
    while frontier:
        s = frontier.pop()
        if s in reach or not live(s):
            continue
        reach.append(s)
        opp = {b: comp.action(s, b) for b in range(n) if b != agent}
        for x in range(S):
            if s[x] != UNOWNED or x in opp.values():
                continue  # occupied (protocol) or a collision (s_bot)
            nxt = list(s)
            nxt[x] = agent
            for b, j in opp.items():
                nxt[j] = b
            frontier.append(tuple(nxt))
    options = [[i for i in range(S) if s[i] == UNOWNED] for s in reach]
    best = -1
    for choice in itertools.product(*options):
        strategy = dict(zip(reach, choice))
        s = tuple([UNOWNED] * S)
        failed = False
        while live(s):
            prof = [strategy[s] if b == agent else comp.action(s, b) for b in range(n)]
            if len(set(prof)) < n:
                failed = True  # s_bot
                break
            nxt = list(s)
            for b, j in enumerate(prof):
                nxt[j] = b
            s = tuple(nxt)
        if failed:
            continue
        r = rewards_from_owner(s, n, nb)
        if all(r[b] >= demands[b] for b in range(n)):
            best = max(best, r[agent])
    assert best >= 0
    return best


def _payoff_floor_improves(play: Play, agent: int, nb: dict) -> bool:
    """The opponent model of the tool before the revision (working note 01).

    The other agents repeat their recorded actions at the states of the play
    and choose freely at every other state, and a deviation counts only if
    it raises the reward of *agent* without lowering the reward of any other
    agent.  Returns ``True`` if such a deviation exists.
    """
    n = play.num_agents
    S = len(play.states[0])
    m = play.horizon
    cur = rewards_from_owner(play.final_state, n, nb)
    on_play = {s: t for t, s in enumerate(play.states[:-1])}
    others = [b for b in range(n) if b != agent]

    @functools.lru_cache(maxsize=None)
    def improves_from(t: int, s: tuple[int, ...]) -> bool:
        if t == m:
            r = rewards_from_owner(s, n, nb)
            return r[agent] > cur[agent] and all(r[b] >= cur[b] for b in others)
        free = [i for i in range(S) if s[i] == UNOWNED]
        tp = on_play.get(s)
        if tp is not None:
            choices: object = [tuple(play.actions[tp][b] for b in others)]
        else:
            choices = itertools.permutations(free, len(others))
        for oc in choices:  # type: ignore[attr-defined]
            for x in free:
                if x in oc:
                    continue
                nxt = list(s)
                nxt[x] = agent
                for b, j in zip(others, oc):
                    nxt[j] = b
                if improves_from(t + 1, tuple(nxt)):
                    return True
        return False

    return improves_from(0, tuple([UNOWNED] * S))


# ---------------------------------------------------------------------------
# Replay, protocol and phi_sat
# ---------------------------------------------------------------------------


class TestReplay:
    def test_running_example_rewards(self) -> None:
        t = _running_example()
        res = certify(t, 3, 4, (2, 2, 2), FIG_3_1)
        assert res.rewards == (2, 4, 4)  # Example 3.9.1

    def test_collision_rejected(self, territory_2x2: Territory) -> None:
        res = certify(territory_2x2, 2, 2, None, [[0, 0], [1, 2]])
        assert not res.accepted and res.keyword == "collision"
        assert "round 0" in res.reason
        assert res.rewards == () and res.best_responses == ()

    def test_collision_in_last_round(self) -> None:
        res = certify(_t("...", "..."), 2, 3, None, [[0, 1], [2, 3], [4, 4]])
        assert not res.accepted and res.keyword == "collision"
        assert "round 2" in res.reason

    def test_occupied_sector_rejected(self, territory_2x2: Territory) -> None:
        res = certify(territory_2x2, 2, 2, None, [[0, 1], [0, 2]])
        assert not res.accepted and res.keyword == "protocol"

    def test_unknown_sector_rejected(self, territory_2x2: Territory) -> None:
        for bad in (4, -1, Coord(5, 5), "x", True):
            res = certify(territory_2x2, 2, 2, None, [[0, 1], [2, bad]])
            assert not res.accepted and res.keyword == "invalid", bad

    def test_shape_of_the_play(self, territory_2x2: Territory) -> None:
        cases = [
            (2, 3, None, [[0, 1], [2, 3]]),  # horizon is not floor(|T|/n)
            (2, 2, None, [[0, 1]]),  # one round short
            (2, 2, None, [[0, 1], [2, 3], [0, 1]]),  # one round too many
            (2, 2, None, [[0, 1], [2]]),  # one action short
            (2, 2, (1,), [[0, 1], [2, 3]]),  # demands of the wrong length
            (2, 2, (1, -1), [[0, 1], [2, 3]]),
            (0, 0, None, []),
            (2, 2, None, [0, 1]),  # not a list of profiles
            (2, 2, None, None),
        ]
        for n, h, d, acts in cases:
            res = certify(territory_2x2, n, h, d, acts)
            assert not res.accepted and res.keyword == "invalid", (n, h, d, acts)

    def test_horizon_none_selects_m(self, territory_2x2: Territory) -> None:
        res = certify(territory_2x2, 2, None, None, [[0, 1], [2, 3]])
        assert res.accepted

    def test_below_demand_rejected(self) -> None:
        t = _running_example()
        # Agent a has reward 2 in Figure 3.1, below a demand of 3.
        res = certify(t, 3, 4, (3, 2, 2), FIG_3_1)
        assert not res.accepted and res.keyword == "unsatisfying"
        assert res.rewards == (2, 4, 4) and res.best_responses == ()

    def test_coordinates_and_indices_agree(self) -> None:
        t = _running_example()
        sectors = t.ordered_sectors()
        coords = [[sectors[i] for i in prof] for prof in FIG_3_1]
        pairs = [[(c.x, c.y) for c in prof] for prof in coords]
        dicts = [[{"x": c.x, "y": c.y} for c in prof] for prof in coords]
        ref = certify(t, 3, 4, (2, 2, 2), FIG_3_1)
        for acts in (coords, pairs, dicts):
            assert certify(t, 3, 4, (2, 2, 2), acts) == ref

    def test_no_rounds(self) -> None:
        t = _t("..")
        assert certify(t, 3, 0, None, []).accepted  # m = 0, rewards 0
        res = certify(t, 3, 0, (0, 1, 0), [])
        assert not res.accepted and res.keyword == "unsatisfying"

    def test_single_agent(self) -> None:
        t = _t("..#", "...")
        acts = [[0], [1], [2], [3], [4]]
        res = certify(t, 1, 5, None, acts)
        assert res.accepted and res.rewards == (5,)

    def test_rewards_match_definition_3_9_2(self) -> None:
        rng = random.Random(4)
        for rows, n in [(("...", "...", "..."), 3), (("..#..", "....."), 2)]:
            t = _t(*rows)
            nb = t.neighbor_map()
            for _ in range(30):
                play = random_play(len(t), n, len(t) // n, rng)
                res = certify(t, n, None, None, play.actions)
                assert res.rewards == rewards_from_owner(play.final_state, n, nb)


# ---------------------------------------------------------------------------
# Best responses against the canonical completion
# ---------------------------------------------------------------------------


TINY_STRATEGY_CASES = [
    (("..", ".."), 2, None),
    (("..", ".."), 2, (2, 1)),
    (("....",), 2, None),
    (("...", "..."), 3, None),
    (("..", "..", ".."), 3, (1, 1, 1)),
    ((".#.", "..."), 2, None),
]

SEQUENCE_CASES = [
    (("...", "..."), 2, None),
    (("...", "..."), 2, (2, 2)),
    (("...", "...", "..."), 2, None),
    (("...", "...", "..."), 3, None),
    (("...", "...", "..."), 3, (2, 1, 2)),
    (("..#", "...", "..."), 2, (2, 2)),
    (("....", "...."), 4, None),
    (("..#..", "..#.."), 2, None),
    (("..", "..", ".."), 2, (3, 1)),
    (("...", "..#"), 3, None),  # n does not divide |T|
]


class TestBestResponses:
    @pytest.mark.parametrize("rows,n,demands", TINY_STRATEGY_CASES)
    def test_matches_strategy_enumeration(self, rows, n, demands) -> None:
        t = _t(*rows)
        d = demands or (0,) * n
        nb = t.neighbor_map()
        rng = random.Random(len(t) * 10 + n + sum(d))
        checked = 0
        for _ in range(20):
            play = random_play(len(t), n, len(t) // n, rng)
            cur = rewards_from_owner(play.final_state, n, nb)
            if any(cur[b] < d[b] for b in range(n)):
                continue
            res = certify(t, n, None, demands, play.actions)
            expected = tuple(_strategy_enumeration_br(t, play, a, d) for a in range(n))
            assert res.best_responses == expected, play.actions
            assert res.accepted == all(expected[a] <= cur[a] for a in range(n))
            checked += 1
        assert checked > 0

    @pytest.mark.parametrize("rows,n,demands", SEQUENCE_CASES)
    def test_matches_sequence_enumeration(self, rows, n, demands) -> None:
        t = _t(*rows)
        d = demands or (0,) * n
        nb = t.neighbor_map()
        rng = random.Random(len(t) * 10 + n)
        checked = accepted = 0
        for _ in range(25):
            play = random_play(len(t), n, len(t) // n, rng)
            cur = rewards_from_owner(play.final_state, n, nb)
            res = certify(t, n, None, demands, play.actions)
            if any(cur[b] < d[b] for b in range(n)):
                assert res.keyword == "unsatisfying"
                continue
            comp = CanonicalCompletion(play)
            expected = tuple(best_deviation(comp, a, nb, d)[0] for a in range(n))
            assert res.best_responses == expected, play.actions
            assert res.rewards == cur
            ne = all(expected[a] <= cur[a] for a in range(n))
            assert res.accepted == ne
            assert (res.reason == "ok") == ne
            if not ne:
                assert res.keyword == "deviation"
            checked += 1
            accepted += ne
        assert checked > 0

    def test_known_equilibria_and_counts(self) -> None:
        # Every agent at reward m needs no search (Lemma 3.10.1).
        t = _t("...", "...")
        res = certify(t, 2, 3, None, [[0, 3], [1, 4], [2, 5]])
        assert res.accepted and res.best_responses == (3, 3)
        assert res.states_explored == 0

    def test_reentry_into_the_play(self) -> None:
        # A deviation that returns to a state of the play meets the recorded
        # actions there again; the oracle covers this, and so must the search.
        t = _t("....", "....")
        rng = random.Random(11)
        nb = t.neighbor_map()
        for _ in range(40):
            play = random_play(8, 2, 4, rng)
            comp = CanonicalCompletion(play)
            res = certify(t, 2, 4, None, play.actions)
            assert res.best_responses == tuple(
                best_deviation(comp, a, nb)[0] for a in range(2)
            )


class TestBounds:
    """The pruning test never cuts off a reachable reward."""

    @pytest.mark.parametrize(
        "rows",
        [("....", "....", "...."), ("..#.", "....", ".#.."), (".#..", ".#..", "....")],
    )
    def test_never_prunes_a_reachable_reward(self, rows) -> None:
        t = _t(*rows)
        board = cert_mod._Board.of(t)
        bits = board.bit_of_index
        S = len(t)
        rng = random.Random(len(rows[0]) + sum(r.count(".") for r in rows))
        pruned = 0
        for _ in range(150):
            k = rng.randint(1, 4)
            own_n = rng.randint(0, S - k)
            cells = rng.sample(range(S), own_n + k + rng.randint(0, S - own_n - k))
            own = sum(bits[i] for i in cells[:own_n])
            free_cells = cells[own_n:]
            free = sum(bits[i] for i in free_cells)
            # The best reward over every choice of k free sectors, ignoring
            # the opponents, bounds what any continuation reaches.
            best = 0
            for extra in itertools.combinations(free_cells, k):
                mask = own | sum(bits[i] for i in extra)
                best = max(best, board.largest_component(mask))
            for threshold in range(best):
                assert cert_mod._may_exceed(board, own, free, k, threshold)
            assert not cert_mod._may_exceed(board, own, free, k, own_n + k)
            pruned += not cert_mod._may_exceed(board, own, free, k, best)
        assert pruned > 0  # the test is not vacuous

    def test_no_free_sector(self) -> None:
        board = cert_mod._Board.of(_t("..."))
        assert not cert_mod._may_exceed(board, board.bit_of_index[0], 0, 1, 0)


# ---------------------------------------------------------------------------
# The old payoff-floor check
# ---------------------------------------------------------------------------


class TestPayoffFloorPlay:
    """A play that the payoff-floor check accepted but that is not an equilibrium."""

    def test_floor_model_accepts_but_completion_refutes(self) -> None:
        t = _running_example()
        nb = t.neighbor_map()
        play = Play.from_actions(12, 3, OLD_SIBIS_PLAY)
        assert play.rewards(nb) == (4, 2, 4)
        # Under the old opponent model no agent can improve without hurting
        # another agent, so IBIS reported an equilibrium.
        assert not any(_payoff_floor_improves(play, a, nb) for a in range(3))
        # Against the canonical completion agent b (tool agent 1) improves.
        for demands in (None, (2, 2, 2)):
            res = certify(t, 3, 4, demands, OLD_SIBIS_PLAY)
            assert not res.accepted and res.keyword == "deviation"
            assert "agent 1" in res.reason
            comp = CanonicalCompletion(play)
            d = demands or (0, 0, 0)
            assert res.best_responses == tuple(
                best_deviation(comp, a, nb, d)[0] for a in range(3)
            )
            assert res.best_responses[1] == 3 > res.rewards[1]


# ---------------------------------------------------------------------------
# Independence from the rest of the tool
# ---------------------------------------------------------------------------


class TestIndependence:
    def test_uses_no_model_or_strategy_code(self, monkeypatch) -> None:
        import rcmas.model as model
        import rcmas.strategy as strategy

        def boom(*_a, **_k):
            raise AssertionError("the certifier must not call this")

        for name in (
            "rewards_from_owner",
            "largest_component_weight",
            "largest_region_size",
            "connected_components",
            "scores",
            "evolve",
        ):
            monkeypatch.setattr(model, name, boom)
        for name in (
            "default_action",
            "free_sector",
            "is_live",
            "canonical_completion",
        ):
            monkeypatch.setattr(strategy, name, boom)
        monkeypatch.setattr(strategy, "CanonicalCompletion", boom)
        t = _running_example()
        res = certify(t, 3, 4, None, OLD_SIBIS_PLAY)
        assert res.keyword == "deviation" and res.states_explored > 0

    def test_no_z3(self) -> None:
        code = (
            "import sys\n"
            "from rcmas.model import Territory\n"
            "from rcmas.certify import certify\n"
            "t = Territory.from_ascii(['...', '...'])\n"
            "r = certify(t, 2, 3, None, [[0, 1], [2, 3], [4, 5]])\n"
            "assert r.reason, r\n"
            "assert 'z3' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_own_enumeration(self) -> None:
        """A wrong enumeration in the territory changes only the index meaning.

        The certifier derives Remark 3.1.1 from the coordinates, so the same
        play given by coordinates gets the same verdict whatever order the
        territory object lists its sectors in.
        """
        good = _running_example()
        order = tuple(reversed(good.ordered_sectors()))
        bad = Territory(good.sectors, order, {c: i for i, c in enumerate(order)})
        sectors = good.ordered_sectors()
        for acts in (FIG_3_1, OLD_SIBIS_PLAY):
            coords = [[sectors[i] for i in prof] for prof in acts]
            via_bad = [[bad.index_of(c) for c in prof] for prof in coords]
            ref = certify(good, 3, 4, None, acts)
            assert certify(bad, 3, 4, None, coords) == ref
            assert certify(bad, 3, 4, None, via_bad) == ref


class TestDeadline:
    def test_past_deadline(self) -> None:
        t = _running_example()
        res = certify(t, 3, 4, None, FIG_3_1, deadline=time.monotonic() - 1.0)
        assert not res.accepted and res.keyword == "deadline"
        assert isinstance(res, CertResult)

    def test_generous_deadline(self) -> None:
        t = _running_example()
        ref = certify(t, 3, 4, None, FIG_3_1)
        res = certify(t, 3, 4, None, FIG_3_1, deadline=time.monotonic() + 60.0)
        assert res == ref


# ---------------------------------------------------------------------------
# Equilibria returned by the synthesisers
# ---------------------------------------------------------------------------


SYNTH_CASES = [
    (("...", "...", "..."), 2, None),
    (("..", "..", ".."), 3, None),
    (("....", "...."), 2, (2, 2)),
    (("..#", "...", "..."), 2, (2, 1)),
]


def _check_returned(res, t: Territory, n: int, demands) -> CertResult | None:
    """Certify the play a synthesiser returned; an equilibrium must be accepted."""
    if res.play is None:
        assert not res.found_ne
        return None
    out = certify(t, n, None, demands, res.play.actions)
    assert out.rewards == tuple(res.payoff_by_agent)
    if res.found_ne:
        assert out.accepted, out
        assert out.best_responses == out.rewards
    return out


class TestSynthesisers:
    """Every equilibrium that a synthesiser returns passes the certifier.

    On the running-example territory (``interesting.txt``) with two agents
    every synthesiser returns an equilibrium in which an agent has reward
    below ``m``, so the certifier has to search its deviations.
    """

    @pytest.mark.parametrize("rows,n,demands", SYNTH_CASES)
    def test_ibis_and_sibis(self, rows, n, demands) -> None:
        pytest.importorskip("z3")
        from rcmas.ibis import solve_ibis, solve_sibis

        t = _t(*rows)
        found = 0
        for solve in (solve_ibis, solve_sibis):
            for symmetry in (False, True):
                res = solve(
                    territory=t, num_agents=n, demands=demands, symmetry=symmetry
                )
                _check_returned(res, t, n, demands)
                found += res.found_ne
        assert found > 0

    def test_running_example_territory(self) -> None:
        pytest.importorskip("z3")
        from rcmas.cegar import solve_cegar
        from rcmas.ibis import solve_ibis, solve_sibis
        from rcmas.qibis import QibisConfig, solve_qibis

        t = _running_example()
        cfg = QibisConfig(offline_episodes=30, online_episodes=30)
        runs = {
            "ibis": solve_ibis(territory=t, num_agents=2),
            "sibis": solve_sibis(territory=t, num_agents=2),
            "qibis": solve_qibis(territory=t, num_agents=2, cfg=cfg),
            "cegar": solve_cegar(territory=t, num_agents=2, synthesiser="sibis"),
        }
        searched = 0
        for name, res in runs.items():
            assert res.found_ne, name
            out = _check_returned(res, t, 2, None)
            assert out is not None
            searched += out.states_explored > 0
        # Which equilibria Z3 leads to varies, but on this territory the
        # synthesisers return equilibria below m (here (4, 6) or (5, 6)).
        assert searched > 0

    def test_running_example_three_agents(self) -> None:
        pytest.importorskip("z3")
        from rcmas.ibis import solve_sibis

        t = _running_example()
        res = solve_sibis(territory=t, num_agents=3)
        out = _check_returned(res, t, 3, None)
        if res.found_ne:
            assert out is not None and out.states_explored > 0

    @pytest.mark.parametrize("rows,n,demands", SYNTH_CASES[:3])
    def test_qibis(self, rows, n, demands) -> None:
        pytest.importorskip("z3")
        from rcmas.qibis import QibisConfig, solve_qibis

        t = _t(*rows)
        cfg = QibisConfig(demands=demands, offline_episodes=30, online_episodes=30)
        res = solve_qibis(territory=t, num_agents=n, cfg=cfg)
        _check_returned(res, t, n, demands)

    @pytest.mark.parametrize("rows,n,demands", SYNTH_CASES[:3])
    def test_cegar(self, rows, n, demands) -> None:
        pytest.importorskip("z3")
        from rcmas.cegar import solve_cegar

        t = _t(*rows)
        for partition in ("orbit", "trivial"):
            res = solve_cegar(
                territory=t,
                num_agents=n,
                initial_partition=partition,
                synthesiser="sibis",
                demands=demands,
            )
            _check_returned(res, t, n, demands)


# ---------------------------------------------------------------------------
# rcmas certify --results
# ---------------------------------------------------------------------------


def _record(name: str, grid: str, n: int, play, **extra) -> dict:
    rec = {
        "suite": "test",
        "name": name,
        "grid_path": grid,
        "num_agents": n,
        "horizon": None,
        "algorithm": "sibis",
        "symmetry": False,
        "run": 1,
        "outcome": "ne",
        "play": play,
    }
    rec.update(extra)
    return rec


class TestCli:
    def _write(self, tmp_path: Path, records: list[dict]) -> Path:
        path = tmp_path / "results.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return path

    def test_mixed_log(self, tmp_path: Path, capsys) -> None:
        from rcmas.cli import main

        # A grid path from another machine resolves to the tool's grids/.
        moved = "/elsewhere/rcmas/experiments/../grids/asymmetric/interesting.txt"
        g33 = str(GRIDS / "symmetric" / "3x3.txt")
        good_33 = [[0, 8], [1, 7], [2, 6], [3, 5]]
        records = [
            _record("fig", moved, 3, FIG_3_1, demands=[2, 2, 2], payoff=[2, 4, 4]),
            _record("old", moved, 3, OLD_SIBIS_PLAY, payoff=[4, 2, 4]),
            _record("nop", g33, 2, None),
            _record("ok33", g33, 2, good_33, payoff=[4, 4]),
            _record("wrong-payoff", g33, 2, good_33, payoff=[4, 3]),
            _record("nc", g33, 2, good_33, outcome="n/c"),
            {
                "name": "legacy",
                "grid_path": g33,
                "num_agents": 2,
                "horizon": 4,
                "found_ne": True,
                "payoff": [4, 4],
                "play": good_33,
            },
        ]
        path = self._write(tmp_path, records)
        out_path = tmp_path / "cert.jsonl"
        rc = main(["certify", "--results", str(path), "--output", str(out_path)])
        text = capsys.readouterr().out
        assert rc == 1
        assert "6 equilibria: 2 accepted, 3 rejected, 1 unchecked" in text
        assert "rejected  old" in text and "deviation" in text
        assert "rejected  wrong-payoff" in text and "rewards" in text
        assert "rejected  fig" in text
        rows = [json.loads(line) for line in open(out_path, encoding="utf-8")]
        by_name = {r["name"]: r for r in rows}
        assert by_name["ok33"]["status"] == "accepted"
        assert by_name["ok33"]["best_responses"] == [4, 4]
        assert by_name["old"]["best_responses"][1] == 3
        assert by_name["nop"]["status"] == "unchecked"
        assert "nc" not in by_name

    def test_all_accepted(self, tmp_path: Path) -> None:
        g33 = str(GRIDS / "symmetric" / "3x3.txt")
        path = self._write(
            tmp_path, [_record("ok33", g33, 2, [[0, 8], [1, 7], [2, 6], [3, 5]])]
        )
        buf = io.StringIO()
        assert run_logs([path], stream=buf) == 0
        assert "1 equilibria: 1 accepted, 0 rejected, 0 unchecked" in buf.getvalue()

    def test_missing_grid_and_timeout(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                _record("lost", "/nowhere/else.txt", 2, [[0, 1]]),
                _record(
                    "slow", "asymmetric/interesting.txt", 3, FIG_3_1, demands=[2, 2, 2]
                ),
            ],
        )
        buf = io.StringIO()
        rc = run_logs([path], grid_root=GRIDS, timeout_s=-1.0, stream=buf)
        text = buf.getvalue()
        assert rc == 2
        assert "grid not found" in text and "deadline" in text
