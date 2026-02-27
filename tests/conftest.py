"""Shared test fixtures for RCMAS."""

from __future__ import annotations

import io

import pytest

from rcmas.model import Coord, State, Territory


# ---------------------------------------------------------------------------
# Territories
# ---------------------------------------------------------------------------

@pytest.fixture
def territory_1x1() -> Territory:
    """Single-sector territory."""
    return Territory.from_ascii(["."])


@pytest.fixture
def territory_2x1() -> Territory:
    """Two sectors in a row: (0,0) (1,0)."""
    return Territory.from_ascii([".."])


@pytest.fixture
def territory_2x2() -> Territory:
    """2x2 square: 4 sectors."""
    return Territory.from_ascii(["..", ".."])


@pytest.fixture
def territory_3x3() -> Territory:
    """3x3 square: 9 sectors."""
    return Territory.from_ascii(["...", "...", "..."])


@pytest.fixture
def territory_L_shape() -> Territory:
    """L-shaped territory (5 sectors):
    ..
    .
    ..
    """
    return Territory.from_ascii(["..", ".", ".."])


@pytest.fixture
def territory_disconnected() -> Territory:
    """Two disconnected components (4 sectors):
    ..#..
    """
    return Territory.from_ascii(["..#.."])


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

@pytest.fixture
def state_2x2(territory_2x2: Territory) -> State:
    return State.initial(territory_2x2, num_agents=2)


@pytest.fixture
def state_3x3(territory_3x3: Territory) -> State:
    return State.initial(territory_3x3, num_agents=2)
