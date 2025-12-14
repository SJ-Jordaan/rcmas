from qlearning.engine import Coord, GameState, GameStatus, Territory


def test_collision_causes_defeat():
    territory = Territory.from_ascii([
        "..",
        "..",
    ])
    s0 = GameState.new(territory, num_agents=2)
    coord = Coord(0, 0)
    s1, outcome = s0.step({0: coord, 1: coord})
    assert s1 is s0
    assert outcome.status == GameStatus.DEFEAT


def test_victory_when_all_claimed():
    territory = Territory.from_ascii(["."])
    s0 = GameState.new(territory, num_agents=1)
    s1, outcome = s0.step({0: Coord(0, 0)})
    assert outcome.status == GameStatus.VICTORY
    assert s1.is_terminal()


def test_largest_region_size_uses_4_neighborhood():
    territory = Territory.from_ascii([
        "..",
        "..",
    ])
    s = GameState.new(territory, num_agents=1)
    s, _ = s.step({0: Coord(0, 0)})
    s, _ = s.step({0: Coord(1, 1)})
    # diagonal only => two regions of size 1
    assert s.largest_region_size(0) == 1
    s, _ = s.step({0: Coord(0, 1)})
    # now (0,0)-(0,1)-(1,1) connected => size 3
    assert s.largest_region_size(0) == 3
