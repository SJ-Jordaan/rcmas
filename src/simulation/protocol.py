"""
Protocol for defining which actions are available to agents in each state.
"""

from typing import Set
from domain.state import State
from domain.agent import Agent, DUMMY_AGENT
from domain.action import Action


class ActionAvailabilityProtocol:
    """
    Implements the action availability protocol P: S × Agt → 2^Act.
    
    Defines which actions are available to each agent in each state.
    According to the paper: P(s,a) = {act^a_{i,j} | s(i,j) = a_0}
    
    I.e., an agent can attempt to occupy any currently unoccupied sector.
    """
    
    @staticmethod
    def get_available_actions(state: State, agent: Agent) -> Set[Action]:
        """
        Get all actions available to an agent in a given state.
        
        Args:
            state: The current state
            agent: The agent whose available actions to compute
            
        Returns:
            Set of actions the agent can choose from
        """
        available_actions = set()
        
        # Agent can attempt to occupy any unoccupied sector
        for sector in state.get_unoccupied_sectors():
            available_actions.add(Action(agent=agent, sector=sector))
        
        return available_actions
    
    @staticmethod
    def is_action_available(state: State, action: Action) -> bool:
        """
        Check if a specific action is available in a state.
        
        Args:
            state: The current state
            action: The action to check
            
        Returns:
            True if the action is available, False otherwise
        """
        # Action is available if the target sector is unoccupied
        return state.is_unoccupied(action.sector)
    
    @staticmethod
    def get_all_available_actions(state: State, agents: Set[Agent]) -> dict[Agent, Set[Action]]:
        """
        Get available actions for all agents in a state.
        
        Args:
            state: The current state
            agents: Set of all agents
            
        Returns:
            Dictionary mapping each agent to their available actions
        """
        return {
            agent: ActionAvailabilityProtocol.get_available_actions(state, agent)
            for agent in agents
        }
