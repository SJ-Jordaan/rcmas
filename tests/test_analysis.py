"""
Unit tests for cohesive region analysis.
"""

import pytest
from domain import Territory, Agent, State, Sector
from analysis import CohesiveRegionAnalyzer, RegionMetrics


class TestCohesiveRegionAnalyzer:
    """Tests for cohesive region computation."""
    
    def test_single_cohesive_region(self):
        """Test detection of a single connected region."""
        territory = Territory(5, 5)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=4)
        
        # Create a connected region
        state.set_occupant(Sector(1, 1), agent)
        state.set_occupant(Sector(1, 2), agent)
        state.set_occupant(Sector(2, 1), agent)
        state.set_occupant(Sector(2, 2), agent)
        
        analyzer = CohesiveRegionAnalyzer(state)
        regions = analyzer.get_all_cohesive_regions(agent)
        
        assert len(regions) == 1
        assert len(regions[0]) == 4
    
    def test_multiple_disconnected_regions(self):
        """Test detection of multiple disconnected regions."""
        territory = Territory(5, 5)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=4)
        
        # Create two disconnected regions
        state.set_occupant(Sector(1, 1), agent)
        state.set_occupant(Sector(1, 2), agent)
        
        state.set_occupant(Sector(4, 4), agent)
        state.set_occupant(Sector(4, 5), agent)
        state.set_occupant(Sector(5, 4), agent)
        
        analyzer = CohesiveRegionAnalyzer(state)
        regions = analyzer.get_all_cohesive_regions(agent)
        
        assert len(regions) == 2
        region_sizes = [len(r) for r in regions]
        assert 2 in region_sizes
        assert 3 in region_sizes
    
    def test_largest_region_selection(self):
        """Test finding the largest cohesive region."""
        territory = Territory(5, 5)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=4)
        
        # Small region
        state.set_occupant(Sector(1, 1), agent)
        
        # Larger region
        for i in range(3, 6):
            for j in range(3, 5):
                state.set_occupant(Sector(i, j), agent)
        
        analyzer = CohesiveRegionAnalyzer(state)
        largest = analyzer.get_largest_cohesive_region(agent)
        
        assert len(largest) == 6
    
    def test_objective_checking(self):
        """Test objective satisfaction checking."""
        territory = Territory(5, 5)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=4)
        
        analyzer = CohesiveRegionAnalyzer(state)
        
        # Initially no regions
        assert not analyzer.check_objective_met(agent)
        
        # Add connected sectors
        for i in range(1, 5):
            state.set_occupant(Sector(1, i), agent)
        
        # Re-create analyzer with updated state
        analyzer = CohesiveRegionAnalyzer(state)
        assert analyzer.check_objective_met(agent)
    
    def test_fragmentation_score(self):
        """Test fragmentation score computation."""
        territory = Territory(5, 5)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=4)
        
        # Single cohesive region (low fragmentation)
        for i in range(1, 5):
            state.set_occupant(Sector(1, i), agent)
        
        analyzer1 = CohesiveRegionAnalyzer(state)
        frag1 = analyzer1.get_fragmentation_score(agent)
        
        # Multiple small regions (high fragmentation)
        state2 = State.create_initial_state(territory)
        state2.set_occupant(Sector(1, 1), agent)
        state2.set_occupant(Sector(3, 3), agent)
        state2.set_occupant(Sector(5, 5), agent)
        
        analyzer2 = CohesiveRegionAnalyzer(state2)
        frag2 = analyzer2.get_fragmentation_score(agent)
        
        # More regions = higher fragmentation
        assert frag2 > frag1
    
    def test_perimeter_sectors(self):
        """Test perimeter/expansion opportunity detection."""
        territory = Territory(5, 5)
        state = State.create_initial_state(territory)
        agent = Agent(id="a1", objective=4)
        
        # Create a 2x2 region
        region = set()
        for i in range(1, 3):
            for j in range(1, 3):
                sector = Sector(i, j)
                state.set_occupant(sector, agent)
                region.add(sector)
        
        analyzer = CohesiveRegionAnalyzer(state)
        perimeter = analyzer.get_perimeter_sectors(region)
        
        # Should have unoccupied neighbors
        assert len(perimeter) > 0
        # All perimeter sectors should be unoccupied
        for sector in perimeter:
            assert state.is_unoccupied(sector)


class TestRegionMetrics:
    """Tests for region metrics computation."""
    
    def test_compute_all_metrics(self):
        """Test comprehensive metrics computation."""
        territory = Territory(5, 5)
        state = State.create_initial_state(territory)
        
        agent1 = Agent(id="a1", objective=3)
        agent2 = Agent(id="a2", objective=4)
        
        # Agent1: single region of 3
        for i in range(1, 4):
            state.set_occupant(Sector(1, i), agent1)
        
        # Agent2: two regions (2 and 3)
        state.set_occupant(Sector(3, 1), agent2)
        state.set_occupant(Sector(3, 2), agent2)
        
        for i in range(5, 8):
            if i <= 5:
                state.set_occupant(Sector(5, i-4), agent2)
        
        metrics = RegionMetrics.compute_all_metrics(state, [agent1, agent2])
        
        assert agent1 in metrics
        assert agent2 in metrics
        
        # Check agent1 metrics
        assert metrics[agent1]['max_region_size'] == 3
        assert metrics[agent1]['objective_met'] == True
        
    def test_winner_determination(self):
        """Test winner determination."""
        territory = Territory(5, 5)
        state = State.create_initial_state(territory)
        
        agent1 = Agent(id="a1", objective=3)
        agent2 = Agent(id="a2", objective=3)
        
        # Agent1 has larger region
        for i in range(1, 6):
            state.set_occupant(Sector(1, i), agent1)
        
        # Agent2 has smaller region
        for i in range(1, 4):
            state.set_occupant(Sector(3, i), agent2)
        
        winner = RegionMetrics.get_winner(state, [agent1, agent2])
        assert winner == agent1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
