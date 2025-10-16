"""
Abstract base class for agent strategies using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Set

from domain.state import State
from domain.agent import Agent
from domain.action import Action


class AbstractStrategy(ABC):
    """
    Abstract base class for agent strategies α: S → Act.
    
    A strategy determines which action an agent will choose in a given state
    from the set of available actions.
    
    This implements the Strategy design pattern, allowing different
    decision-making approaches to be used interchangeably.
    """
    
    @abstractmethod
    def select_action(
        self,
        state: State,
        agent: Agent,
        available_actions: Set[Action]
    ) -> Action:
        """
        Select an action for the agent in the given state.
        
        Args:
            state: The current state of the system
            agent: The agent making the decision
            available_actions: Set of actions available to the agent
            
        Returns:
            The chosen action
            
        Raises:
            ValueError: If no actions are available
        """
        pass
    
    def on_simulation_start(self, state: State, agent: Agent) -> None:
        """
        Called when a simulation starts (optional hook for stateful strategies).
        
        Args:
            state: The initial state
            agent: The agent using this strategy
        """
        pass
    
    def on_simulation_end(
        self,
        final_state: State,
        agent: Agent,
        success: bool
    ) -> None:
        """
        Called when a simulation ends (optional hook for learning strategies).
        
        Args:
            final_state: The final state reached
            agent: The agent using this strategy
            success: Whether the mission succeeded (no collision)
        """
        pass
    
    def on_round_complete(
        self,
        old_state: State,
        action: Action,
        new_state: State,
        agent: Agent
    ) -> None:
        """
        Called after each round completes (optional hook for learning strategies).
        
        Args:
            old_state: State before the action
            action: Action that was taken
            new_state: State after the action
            agent: The agent using this strategy
        """
        pass
    
    def get_name(self) -> str:
        """
        Get a human-readable name for this strategy.
        
        Returns:
            Strategy name
        """
        return self.__class__.__name__
    
    def __str__(self) -> str:
        return self.get_name()
    
    def __repr__(self) -> str:
        return f"{self.get_name()}()"
