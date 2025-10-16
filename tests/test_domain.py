"""
Unit tests for domain models.
"""

import pytest
from domain import (
    Agent, Territory, Sector, State, Action, ActionProfile,
    TerritoryBuilder, ActionProfileBuilder, DUMMY_AGENT
)


class TestSector:
    """Tests for Sector class."""
    
    def test_sector_creation(self):
        sector = Sector(1, 2)
        assert sector.i == 1
        assert sector.j == 2
    
    def test_sector_equality(self):
        s1 = Sector(1, 2)
        s2 = Sector(1, 2)
        s3 = Sector(2, 1)
        assert s1 == s2
        assert s1 != s3
    
    def test_sector_hashable(self):
        s1 = Sector(1, 2)
        s2 = Sector(1, 2)
        sector_set = {s1, s2}
        assert len(sector_set) == 1
    
    def test_adjacent_sectors(self):
        sector = Sector(5, 5)
        adjacent = sector.get_adjacent_sectors()
        assert len(adjacent) == 4
        assert Sector(6, 5) in adjacent
        assert Sector(4, 5) in adjacent
        assert Sector(5, 6) in adjacent
        assert Sector(5, 4) in adjacent


class TestTerritory:
    """Tests for Territory class."""
    
    def test_territory_creation(self):
        territory = Territory(5, 5)
        assert territory.width == 5
        assert territory.height == 5
        assert territory.size() == 25
    
    def test_territory_with_obstacles(self):
        obstacles = {Sector(1, 1), Sector(2, 2)}
        territory = Territory(5, 5, obstacles)
        assert territory.size() == 23
        assert not territory.contains(Sector(1, 1))
        assert territory.contains(Sector(3, 3))
    
    def test_territory_builder(self):
        territory = (TerritoryBuilder(10, 10)
                    .add_obstacle(5, 5)
                    .add_obstacle_region(1, 1, 2, 2)
                    .build())
        assert not territory.contains(Sector(5, 5))
        assert not territory.contains(Sector(1, 1))
        assert territory.size() < 100


class TestAgent:
    """Tests for Agent class."""
    
    def test_agent_creation(self):
        agent = Agent(id="a1", objective=5)
        assert agent.id == "a1"
        assert agent.objective == 5
    
    def test_agent_equality(self):
        a1 = Agent(id="a1", objective=5)
        a2 = Agent(id="a1", objective=3)
        a3 = Agent(id="a2", objective=5)
        assert a1 == a2  # Same id
        assert a1 != a3
    
    def test_agent_hashable(self):
        a1 = Agent(id="a1", objective=5)
        a2 = Agent(id="a1", objective=5)
        agent_set = {a1, a2}
        assert len(agent_set) == 1
    
    def test_dummy_agent(self):
        assert DUMMY_AGENT.id == "a0"
        assert DUMMY_AGENT.objective == 0


class TestState:
    """Tests for State class."""
    
    def test_initial_state(self):
        territory = Territory(3, 3)
        state = State.create_initial_state(territory)
        
        for sector in territory.sectors:
            assert state.is_unoccupied(sector)
            assert state.get_occupant(sector) == DUMMY_AGENT
    
    def test_state_occupancy(self):
        territory = Territory(3, 3)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=3)
        
        sector = Sector(1, 1)
        state.set_occupant(sector, agent)
        
        assert state.is_occupied(sector)
        assert state.get_occupant(sector) == agent
    
    def test_get_sectors_for_agent(self):
        territory = Territory(3, 3)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=3)
        
        state.set_occupant(Sector(1, 1), agent)
        state.set_occupant(Sector(1, 2), agent)
        
        sectors = state.get_sectors_for_agent(agent)
        assert len(sectors) == 2
        assert Sector(1, 1) in sectors
        assert Sector(1, 2) in sectors


class TestAction:
    """Tests for Action and ActionProfile."""
    
    def test_action_creation(self):
        agent = Agent(id="a1", objective=3)
        sector = Sector(1, 1)
        action = Action(agent=agent, sector=sector)
        
        assert action.agent == agent
        assert action.sector == sector
    
    def test_action_profile_builder(self):
        agent1 = Agent(id="a1", objective=3)
        agent2 = Agent(id="a2", objective=3)
        
        builder = ActionProfileBuilder()
        profile = (builder
                  .add_action_by_sector(agent1, Sector(1, 1))
                  .add_action_by_sector(agent2, Sector(2, 2))
                  .build())
        
        assert len(profile) == 2
        assert profile.get_action(agent1).sector == Sector(1, 1)
        assert profile.get_action(agent2).sector == Sector(2, 2)
    
    def test_action_profile_collision_detection(self):
        agent1 = Agent(id="a1", objective=3)
        agent2 = Agent(id="a2", objective=3)
        
        # No collision
        builder = ActionProfileBuilder()
        profile1 = (builder
                   .add_action_by_sector(agent1, Sector(1, 1))
                   .add_action_by_sector(agent2, Sector(2, 2))
                   .build())
        assert not profile1.has_collision()
        
        # Collision
        builder.reset()
        profile2 = (builder
                   .add_action_by_sector(agent1, Sector(1, 1))
                   .add_action_by_sector(agent2, Sector(1, 1))
                   .build())
        assert profile2.has_collision()
        assert Sector(1, 1) in profile2.get_colliding_sectors()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
