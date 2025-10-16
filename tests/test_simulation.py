"""
Unit tests for simulation engine.
"""

import pytest
from domain import Territory, Agent, State, Sector
from simulation import (
    RCMASSimulator, ActionAvailabilityProtocol, Evolution,
    TerminationReason
)
from strategies import RandomStrategy
from domain import ActionProfileBuilder


class TestActionAvailabilityProtocol:
    """Tests for action availability protocol."""
    
    def test_available_actions_initial_state(self):
        territory = Territory(3, 3)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=3)
        
        protocol = ActionAvailabilityProtocol()
        actions = protocol.get_available_actions(state, agent)
        
        # All sectors should be available
        assert len(actions) == 9
    
    def test_available_actions_partial_occupancy(self):
        territory = Territory(3, 3)
        state = State.create_initial_state(territory)
        agent1 = Agent(id="a1", objective=3)
        agent2 = Agent(id="a2", objective=3)
        
        # Occupy some sectors
        state.set_occupant(Sector(1, 1), agent1)
        state.set_occupant(Sector(1, 2), agent1)
        
        protocol = ActionAvailabilityProtocol()
        actions = protocol.get_available_actions(state, agent2)
        
        # Only unoccupied sectors available
        assert len(actions) == 7


class TestEvolution:
    """Tests for evolution relation."""
    
    def test_successful_occupation(self):
        territory = Territory(3, 3)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=3)
        
        # Create action profile with single action
        builder = ActionProfileBuilder()
        profile = builder.add_action_by_sector(agent, Sector(1, 1)).build()
        
        # Apply evolution
        evolution = Evolution()
        next_state = evolution.apply(state, profile)
        
        assert next_state.is_occupied(Sector(1, 1))
        assert next_state.get_occupant(Sector(1, 1)) == agent
    
    def test_collision_detection(self):
        territory = Territory(3, 3)
        state = State.create_initial_state(territory)
        agent1 = Agent(id="a1", objective=3)
        agent2 = Agent(id="a2", objective=3)
        
        # Both agents try to occupy same sector
        builder = ActionProfileBuilder()
        profile = (builder
                  .add_action_by_sector(agent1, Sector(1, 1))
                  .add_action_by_sector(agent2, Sector(1, 1))
                  .build())
        
        evolution = Evolution()
        next_state = evolution.apply(state, profile)
        
        assert next_state.is_failure()
    
    def test_persistence(self):
        territory = Territory(3, 3)
        state = State.create_initial_state(territory)
        agent1 = Agent(id="a1", objective=3)
        agent2 = Agent(id="a2", objective=3)
        
        # Agent1 occupies a sector
        state.set_occupant(Sector(1, 1), agent1)
        
        # Agent2 occupies different sector
        builder = ActionProfileBuilder()
        profile = builder.add_action_by_sector(agent2, Sector(2, 2)).build()
        
        evolution = Evolution()
        next_state = evolution.apply(state, profile)
        
        # Agent1's sector should persist
        assert next_state.get_occupant(Sector(1, 1)) == agent1
        assert next_state.get_occupant(Sector(2, 2)) == agent2
    
    def test_terminal_state_detection(self):
        territory = Territory(2, 2)
        state = State.create_initial_state(territory)
        
        evolution = Evolution()
        assert not evolution.is_terminal(state)
        
        # Occupy all sectors
        agent = Agent(id="a1", objective=2)
        for sector in territory.sectors:
            state.set_occupant(sector, agent)
        
        assert evolution.is_terminal(state)


class TestRCMASSimulator:
    """Tests for main simulator."""
    
    def test_simulator_initialization(self):
        territory = Territory(5, 5)
        agents = [
            Agent(id="a1", objective=3, strategy=RandomStrategy(seed=42)),
            Agent(id="a2", objective=3, strategy=RandomStrategy(seed=43))
        ]
        
        simulator = RCMASSimulator(territory, agents)
        assert simulator.territory == territory
        assert len(simulator.agents) == 2
    
    def test_simulation_runs_to_completion(self):
        territory = Territory(3, 3)
        agents = [
            Agent(id="a1", objective=2, strategy=RandomStrategy(seed=42)),
            Agent(id="a2", objective=2, strategy=RandomStrategy(seed=43))
        ]
        
        simulator = RCMASSimulator(territory, agents, max_rounds=100)
        result = simulator.run()
        
        # Should terminate (either success or collision)
        assert result.termination_reason in [
            TerminationReason.FULLY_OCCUPIED,
            TerminationReason.COLLISION,
            TerminationReason.MAX_ROUNDS_REACHED
        ]
        assert len(result.trajectory) > 0
    
    def test_multiple_runs(self):
        territory = Territory(3, 3)
        agents = [
            Agent(id="a1", objective=2, strategy=RandomStrategy()),
            Agent(id="a2", objective=2, strategy=RandomStrategy())
        ]
        
        simulator = RCMASSimulator(territory, agents, max_rounds=50)
        results = simulator.run_multiple(num_runs=5)
        
        assert len(results) == 5
        for result in results:
            assert result.total_rounds > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
