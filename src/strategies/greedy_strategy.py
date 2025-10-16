"""
Greedy strategy that expands the largest cohesive region.
"""

from typing import Set
from domain.state import State
from domain.agent import Agent
from domain.action import Action
from strategies.abstract_strategy import AbstractStrategy


class GreedyExpansionStrategy(AbstractStrategy):
    """
    Greedy strategy that attempts to expand the largest cohesive region.
    
    In each state, this strategy:
    1. Finds all cohesive regions controlled by the agent
    2. Identifies the largest one
    3. Selects an action that adds an adjacent sector to that region
    
    If no expansion is possible, selects randomly from available actions.
    """
    
    def __init__(self, tie_break_random: bool = True, seed: int = None):
        """
        Initialize the greedy expansion strategy.
        
        Args:
            tie_break_random: If True, break ties randomly; else use first found
            seed: Random seed for tie-breaking (if tie_break_random=True)
        """
        self.tie_break_random = tie_break_random
        self.seed = seed
        
        if tie_break_random:
            import random
            self.rng = random.Random(seed)
        else:
            self.rng = None
    
    def select_action(
        self,
        state: State,
        agent: Agent,
        available_actions: Set[Action]
    ) -> Action:
        """
        Select an action that expands the largest cohesive region.
        
        Args:
            state: Current state
            agent: The agent making the decision
            available_actions: Set of available actions
            
        Returns:
            Action that best expands the largest region
            
        Raises:
            ValueError: If no actions are available
        """
        if not available_actions:
            raise ValueError(f"No available actions for agent {agent.id}")
        
        # Import here to avoid circular dependency
        from analysis.regions import CohesiveRegionAnalyzer
        
        analyzer = CohesiveRegionAnalyzer(state)
        largest_region = analyzer.get_largest_cohesive_region(agent)
        
        if not largest_region:
            # Agent has no sectors yet, pick any action
            return self._select_from_set(available_actions)
        
        # Find actions that are adjacent to the largest region
        expansion_actions = []
        for action in available_actions:
            sector = action.sector
            # Check if this sector is adjacent to any sector in largest region
            for adj_sector in sector.get_adjacent_sectors():
                if adj_sector in largest_region:
                    expansion_actions.append(action)
                    break
        
        if expansion_actions:
            return self._select_from_set(set(expansion_actions))
        else:
            # No direct expansion possible, select any available action
            return self._select_from_set(available_actions)
    
    def _select_from_set(self, actions: Set[Action]) -> Action:
        """Helper to select from a set of actions based on tie-breaking strategy."""
        actions_list = list(actions)
        if self.tie_break_random:
            return self.rng.choice(actions_list)
        else:
            return actions_list[0]
    
    def get_name(self) -> str:
        return "GreedyExpansion"
    
    def __repr__(self) -> str:
        return f"GreedyExpansionStrategy(tie_break_random={self.tie_break_random})"
