"""
Random strategy that selects actions uniformly at random.
"""

import random
from typing import Set

from domain.state import State
from domain.agent import Agent
from domain.action import Action
from strategies.abstract_strategy import AbstractStrategy


class RandomStrategy(AbstractStrategy):
    """
    Strategy that selects actions uniformly at random from available actions.
    
    This serves as a baseline strategy for comparison and testing.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the random strategy.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)
    
    def select_action(
        self,
        state: State,
        agent: Agent,
        available_actions: Set[Action]
    ) -> Action:
        """
        Select a random action from available actions.
        
        Args:
            state: Current state (unused by this strategy)
            agent: The agent making the decision
            available_actions: Set of available actions
            
        Returns:
            A randomly selected action
            
        Raises:
            ValueError: If no actions are available
        """
        if not available_actions:
            raise ValueError(f"No available actions for agent {agent.id}")
        
        return self.rng.choice(list(available_actions))
    
    def get_name(self) -> str:
        return "Random"
    
    def __repr__(self) -> str:
        return f"RandomStrategy(seed={self.seed})"
