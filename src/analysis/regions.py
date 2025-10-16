"""
Cohesive region computation and analysis utilities.
"""

from typing import Set, List, Dict, Optional
from collections import deque

from domain.state import State
from domain.agent import Agent, DUMMY_AGENT
from domain.territory import Sector


class CohesiveRegionAnalyzer:
    """
    Analyzer for computing cohesive regions in RCMAS states.
    
    A cohesive region for an agent is a maximal connected set of sectors
    controlled by that agent, where connectivity is defined by orthogonal
    adjacency (up, down, left, right).
    """
    
    def __init__(self, state: State):
        """
        Initialize analyzer with a state.
        
        Args:
            state: The state to analyze
        """
        self.state = state
        self.territory = state.get_territory()
        self._region_cache: Dict[Agent, List[Set[Sector]]] = {}
    
    def get_cohesive_region(self, start_sector: Sector) -> Set[Sector]:
        """
        Compute the cohesive region starting from a sector.
        
        Uses BFS to find all connected sectors controlled by the same agent.
        Implements the inductive definition from the paper.
        
        Args:
            start_sector: The sector to start from
            
        Returns:
            Set of sectors in the cohesive region
        """
        if start_sector not in self.territory.sectors:
            return set()
        
        agent = self.state.get_occupant(start_sector)
        if agent == DUMMY_AGENT:
            return set()
        
        region: Set[Sector] = set()
        queue = deque([start_sector])
        visited = {start_sector}
        
        while queue:
            current = queue.popleft()
            region.add(current)
            
            # Check all adjacent sectors
            for adj_sector in self.territory.get_adjacent_accessible_sectors(current):
                if adj_sector not in visited:
                    visited.add(adj_sector)
                    # Add to region if controlled by same agent
                    if self.state.get_occupant(adj_sector) == agent:
                        queue.append(adj_sector)
        
        return region
    
    def get_all_cohesive_regions(self, agent: Agent) -> List[Set[Sector]]:
        """
        Get all distinct cohesive regions controlled by an agent.
        
        Args:
            agent: The agent whose regions to find
            
        Returns:
            List of cohesive regions (each region is a set of sectors)
        """
        if agent in self._region_cache:
            return self._region_cache[agent]
        
        agent_sectors = self.state.get_sectors_for_agent(agent)
        if not agent_sectors:
            self._region_cache[agent] = []
            return []
        
        regions: List[Set[Sector]] = []
        unprocessed = agent_sectors.copy()
        
        while unprocessed:
            # Start a new region from an unprocessed sector
            start_sector = next(iter(unprocessed))
            region = self.get_cohesive_region(start_sector)
            regions.append(region)
            
            # Remove all sectors in this region from unprocessed
            unprocessed -= region
        
        self._region_cache[agent] = regions
        return regions
    
    def get_largest_cohesive_region(self, agent: Agent) -> Optional[Set[Sector]]:
        """
        Get the largest cohesive region controlled by an agent.
        
        Args:
            agent: The agent whose largest region to find
            
        Returns:
            The largest cohesive region, or None if agent has no sectors
        """
        regions = self.get_all_cohesive_regions(agent)
        if not regions:
            return None
        return max(regions, key=len)
    
    def get_region_sizes(self, agent: Agent) -> List[int]:
        """
        Get the sizes of all cohesive regions for an agent.
        
        Args:
            agent: The agent to analyze
            
        Returns:
            List of region sizes, sorted in descending order
        """
        regions = self.get_all_cohesive_regions(agent)
        sizes = [len(region) for region in regions]
        return sorted(sizes, reverse=True)
    
    def get_max_region_size(self, agent: Agent) -> int:
        """
        Get the size of the largest cohesive region.
        
        Args:
            agent: The agent to analyze
            
        Returns:
            Size of largest region (0 if no regions)
        """
        largest = self.get_largest_cohesive_region(agent)
        return len(largest) if largest else 0
    
    def check_objective_met(self, agent: Agent) -> bool:
        """
        Check if an agent has met their objective.
        
        Args:
            agent: The agent to check
            
        Returns:
            True if largest region >= objective
        """
        return self.get_max_region_size(agent) >= agent.objective
    
    def get_all_objectives_status(self, agents: List[Agent]) -> Dict[Agent, bool]:
        """
        Check objectives for all agents.
        
        Args:
            agents: List of agents to check
            
        Returns:
            Dictionary mapping agents to whether they met their objective
        """
        return {
            agent: self.check_objective_met(agent)
            for agent in agents
        }
    
    def get_perimeter_sectors(self, region: Set[Sector]) -> Set[Sector]:
        """
        Get the perimeter (boundary) sectors of a region.
        
        Perimeter sectors are unoccupied sectors adjacent to the region.
        
        Args:
            region: The region to analyze
            
        Returns:
            Set of unoccupied sectors adjacent to the region
        """
        perimeter = set()
        
        for sector in region:
            for adj_sector in self.territory.get_adjacent_accessible_sectors(sector):
                if self.state.is_unoccupied(adj_sector):
                    perimeter.add(adj_sector)
        
        return perimeter
    
    def get_expansion_opportunities(self, agent: Agent) -> Dict[Set[Sector], Set[Sector]]:
        """
        Get expansion opportunities for each region of an agent.
        
        Args:
            agent: The agent to analyze
            
        Returns:
            Dictionary mapping each region to its expansion opportunities (perimeter)
        """
        regions = self.get_all_cohesive_regions(agent)
        return {
            frozenset(region): self.get_perimeter_sectors(region)
            for region in regions
        }
    
    def get_fragmentation_score(self, agent: Agent) -> float:
        """
        Compute a fragmentation score for an agent's territories.
        
        Lower score = more cohesive (fewer, larger regions).
        Score = number_of_regions / total_sectors_controlled
        
        Args:
            agent: The agent to analyze
            
        Returns:
            Fragmentation score (0 if no sectors controlled)
        """
        regions = self.get_all_cohesive_regions(agent)
        if not regions:
            return 0.0
        
        num_regions = len(regions)
        total_sectors = sum(len(region) for region in regions)
        
        return num_regions / total_sectors if total_sectors > 0 else 0.0


class RegionMetrics:
    """
    Comprehensive metrics for analyzing region control.
    """
    
    @staticmethod
    def compute_all_metrics(
        state: State,
        agents: List[Agent]
    ) -> Dict[Agent, Dict[str, any]]:
        """
        Compute comprehensive metrics for all agents.
        
        Args:
            state: The state to analyze
            agents: List of agents
            
        Returns:
            Dictionary mapping agents to their metrics
        """
        analyzer = CohesiveRegionAnalyzer(state)
        metrics = {}
        
        for agent in agents:
            agent_metrics = {
                'num_regions': len(analyzer.get_all_cohesive_regions(agent)),
                'region_sizes': analyzer.get_region_sizes(agent),
                'max_region_size': analyzer.get_max_region_size(agent),
                'total_sectors': len(state.get_sectors_for_agent(agent)),
                'objective': agent.objective,
                'objective_met': analyzer.check_objective_met(agent),
                'fragmentation': analyzer.get_fragmentation_score(agent),
            }
            metrics[agent] = agent_metrics
        
        return metrics
    
    @staticmethod
    def get_winner(state: State, agents: List[Agent]) -> Optional[Agent]:
        """
        Determine the winner based on largest cohesive region.
        
        Args:
            state: Final state
            agents: List of agents
            
        Returns:
            Agent with largest region, or None if tie
        """
        analyzer = CohesiveRegionAnalyzer(state)
        max_sizes = {agent: analyzer.get_max_region_size(agent) for agent in agents}
        
        if not max_sizes:
            return None
        
        max_size = max(max_sizes.values())
        winners = [agent for agent, size in max_sizes.items() if size == max_size]
        
        return winners[0] if len(winners) == 1 else None
