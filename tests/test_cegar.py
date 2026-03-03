"""Tests for rcmas.cegar (Algorithm 1)."""

from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from rcmas.abstraction import discrete_partition, orbit_partition
from rcmas.cegar import CegarResult, solve_cegar, verify_ne
from rcmas.model import Coord, Territory


# ---------------------------------------------------------------------------
# TestWeightedSmt — weighted size constraint produces correct values
# ---------------------------------------------------------------------------

class TestWeightedSmt:
    def test_weighted_payoff(self) -> None:
        """Weighted solve on a 2-sector territory with weights (2, 3) should
        produce a payoff of 5 for a single agent owning both."""
        territory = Territory.from_ascii([".."])
        from rcmas.smt_solve import solve_smt_game

        sol = solve_smt_game(
            territory=territory,
            num_agents=1,
            horizon=2,
            objective="sum",
            require_victory=True,
            debug=True,
            weights=(2, 3),
        )
        assert sol.is_sat
        assert sol.payoff_by_agent is not None
        # Agent 0 owns both sectors: weighted size = 2 + 3 = 5
        assert sol.payoff_by_agent[0] == 5

    def test_unit_weights_match_unweighted(self) -> None:
        """Explicit unit weights should produce same payoffs as no weights."""
        territory = Territory.from_ascii(["..", ".."])
        from rcmas.smt_solve import solve_smt_game

        weighted = solve_smt_game(
            territory=territory,
            num_agents=2,
            horizon=2,
            objective="sum",
            require_victory=True,
            debug=True,
            weights=(1, 1, 1, 1),
        )
        unweighted = solve_smt_game(
            territory=territory,
            num_agents=2,
            horizon=2,
            objective="sum",
            require_victory=True,
            debug=True,
        )
        assert weighted.is_sat and unweighted.is_sat
        assert weighted.payoff_by_agent is not None
        assert unweighted.payoff_by_agent is not None
        assert sum(weighted.payoff_by_agent) == sum(unweighted.payoff_by_agent)

    def test_custom_neighbors(self) -> None:
        """Custom neighbors should override default adjacency."""
        # 3 synthetic sectors, only 0~1 adjacent (not 1~2)
        territory = Territory.from_ascii(["..."])
        from rcmas.smt_solve import solve_smt_game

        custom_nb: dict[int, list[int]] = {0: [1], 1: [0], 2: []}
        sol = solve_smt_game(
            territory=territory,
            num_agents=1,
            horizon=3,
            objective="sum",
            require_victory=True,
            debug=True,
            custom_neighbors=custom_nb,
        )
        assert sol.is_sat
        assert sol.payoff_by_agent is not None
        # Agent 0 owns all 3, but sector 2 is disconnected -> max region = 2
        assert sol.payoff_by_agent[0] == 2


# ---------------------------------------------------------------------------
# TestVerifyNe
# ---------------------------------------------------------------------------

class TestVerifyNe:
    def test_known_ne_passes(self, territory_2x1: Territory) -> None:
        """On 2x1 with 2 agents, each gets 1 sector — should be NE."""
        from rcmas.abstraction import LiftedStrategy

        # Each agent owns one sector, payoff 1 each
        sectors = territory_2x1.ordered_sectors()
        lifted = LiftedStrategy(
            terminal_owner=(0, 1),
            actions_by_round=((sectors[0], sectors[1]),),
            payoff_by_agent=(1, 1),
        )
        agent, _, _ = verify_ne(
            territory=territory_2x1,
            num_agents=2,
            horizon=1,
            lifted_strategy=lifted,
        )
        assert agent is None  # No deviating agent -> NE


# ---------------------------------------------------------------------------
# TestCegarSmoke
# ---------------------------------------------------------------------------

class TestCegarSmoke:
    def test_2x1_two_agents(self, territory_2x1: Territory) -> None:
        """2x1 with 2 agents: trivial NE, should find it."""
        result = solve_cegar(
            territory=territory_2x1,
            num_agents=2,
            horizon=1,
        )
        assert result.is_sat
        assert result.found_ne
        assert result.payoff_by_agent is not None

    def test_2x2_two_agents(self, territory_2x2: Territory) -> None:
        """2x2 with 2 agents: should find NE."""
        result = solve_cegar(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
        )
        assert result.is_sat
        assert result.found_ne
        assert result.payoff_by_agent is not None
        assert len(result.payoff_by_agent) == 2

    def test_discrete_partition_2x1(self, territory_2x1: Territory) -> None:
        """Discrete partition should fall through to concrete IBIS."""
        result = solve_cegar(
            territory=territory_2x1,
            num_agents=2,
            horizon=1,
            initial_partition="discrete",
        )
        assert result.is_sat
        assert result.found_ne
        assert result.reason == "concrete_fallback"


# ---------------------------------------------------------------------------
# TestCegarRefinement
# ---------------------------------------------------------------------------

class TestCegarRefinement:
    def test_orbit_partition_triggers_refinement_or_finds_ne(
        self, territory_2x2: Territory
    ) -> None:
        """Orbit partition on 2x2 (1 block) should either find NE or refine."""
        result = solve_cegar(
            territory=territory_2x2,
            num_agents=2,
            horizon=2,
            initial_partition="orbit",
            progress=False,
        )
        # Should eventually find NE (possibly after refinement or concrete fallback)
        assert result.is_sat
        assert result.found_ne

    def test_invalid_synthesiser_raises(self, territory_2x1: Territory) -> None:
        with pytest.raises(ValueError, match="only 'ibis'"):
            solve_cegar(
                territory=territory_2x1,
                num_agents=2,
                horizon=1,
                synthesiser="qibis",
            )

    def test_invalid_partition_type_raises(self, territory_2x1: Territory) -> None:
        with pytest.raises(ValueError, match="unknown partition"):
            solve_cegar(
                territory=territory_2x1,
                num_agents=2,
                horizon=1,
                initial_partition="bogus",
            )

    def test_3x3_finds_ne(self, territory_3x3: Territory) -> None:
        """3x3 with 3 agents and orbit partition (3 blocks) should find NE."""
        result = solve_cegar(
            territory=territory_3x3,
            num_agents=3,
            horizon=3,
            initial_partition="orbit",
        )
        assert result.is_sat
        assert result.found_ne
        assert result.payoff_by_agent is not None
        assert len(result.payoff_by_agent) == 3
