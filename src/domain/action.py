"""
Action and ActionProfile classes for agent actions in RCMAS.
"""

from typing import Dict, Set
from dataclasses import dataclass

from domain.agent import Agent
from domain.territory import Sector


@dataclass(frozen=True)
class Action:
    """
    Represents an action act^a_{i,j} where agent a attempts to occupy sector (i,j).
    
    Immutable value object representing an agent's attempt to control a sector.
    """
    
    agent: Agent
    sector: Sector
    
    def __str__(self) -> str:
        return f"act^{self.agent.id}_{self.sector}"
    
    def __repr__(self) -> str:
        return f"Action({self.agent.id}, {self.sector})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self.agent == other.agent and self.sector == other.sector
    
    def __hash__(self) -> int:
        return hash((self.agent, self.sector))


class ActionProfile:
    """
    Represents an action profile ap: Agt → Act.
    
    Maps each agent to the action it chooses in a given round.
    This is essentially a joint action tuple for all agents.
    """
    
    def __init__(self, actions: Dict[Agent, Action]):
        """
        Initialize an action profile.
        
        Args:
            actions: Dictionary mapping each agent to their chosen action
        """
        self._actions = actions.copy()
        
        # Validate that each action's agent matches the key
        for agent, action in self._actions.items():
            if action.agent != agent:
                raise ValueError(
                    f"Action's agent {action.agent} doesn't match key {agent}"
                )
    
    def get_action(self, agent: Agent) -> Action:
        """
        Get the action chosen by a specific agent.
        
        Args:
            agent: The agent whose action to retrieve
            
        Returns:
            The action chosen by the agent
            
        Raises:
            KeyError: If the agent is not in this profile
        """
        return self._actions[agent]
    
    def get_all_actions(self) -> Dict[Agent, Action]:
        """Returns a copy of all agent-action mappings."""
        return self._actions.copy()
    
    def get_agents(self) -> Set[Agent]:
        """Returns the set of agents in this profile."""
        return set(self._actions.keys())
    
    def get_target_sectors(self) -> Dict[Sector, Set[Agent]]:
        """
        Get sectors targeted by actions and which agents target them.
        
        Useful for detecting collisions (multiple agents targeting the same sector).
        
        Returns:
            Dictionary mapping sectors to the set of agents attempting to occupy them
        """
        sector_targets: Dict[Sector, Set[Agent]] = {}
        
        for agent, action in self._actions.items():
            sector = action.sector
            if sector not in sector_targets:
                sector_targets[sector] = set()
            sector_targets[sector].add(agent)
        
        return sector_targets
    
    def has_collision(self) -> bool:
        """
        Check if multiple agents are attempting to occupy the same sector.
        
        Returns:
            True if there's a collision, False otherwise
        """
        sector_targets = self.get_target_sectors()
        return any(len(agents) > 1 for agents in sector_targets.values())
    
    def get_colliding_sectors(self) -> Set[Sector]:
        """
        Get all sectors that have multiple agents attempting to occupy them.
        
        Returns:
            Set of sectors with collisions
        """
        sector_targets = self.get_target_sectors()
        return {
            sector for sector, agents in sector_targets.items()
            if len(agents) > 1
        }
    
    def __len__(self) -> int:
        """Returns the number of agents in this profile."""
        return len(self._actions)
    
    def __str__(self) -> str:
        action_strs = [f"{agent.id}→{action.sector}" 
                      for agent, action in self._actions.items()]
        return f"ActionProfile({', '.join(action_strs)})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionProfile):
            return NotImplemented
        return self._actions == other._actions


class ActionProfileBuilder:
    """
    Builder pattern for constructing action profiles incrementally.
    """
    
    def __init__(self):
        self._actions: Dict[Agent, Action] = {}
    
    def add_action(self, agent: Agent, action: Action) -> 'ActionProfileBuilder':
        """
        Add an action for an agent.
        
        Args:
            agent: The agent choosing the action
            action: The action chosen
            
        Returns:
            Self for method chaining
        """
        if action.agent != agent:
            raise ValueError(
                f"Action's agent {action.agent} doesn't match provided agent {agent}"
            )
        self._actions[agent] = action
        return self
    
    def add_action_by_sector(
        self,
        agent: Agent,
        sector: Sector
    ) -> 'ActionProfileBuilder':
        """
        Add an action for an agent by specifying the target sector.
        
        Args:
            agent: The agent choosing the action
            sector: The target sector
            
        Returns:
            Self for method chaining
        """
        action = Action(agent=agent, sector=sector)
        self._actions[agent] = action
        return self
    
    def build(self) -> ActionProfile:
        """
        Build and return the action profile.
        
        Returns:
            The constructed ActionProfile
            
        Raises:
            ValueError: If no actions have been added
        """
        if not self._actions:
            raise ValueError("Cannot build empty action profile")
        return ActionProfile(self._actions)
    
    def reset(self) -> 'ActionProfileBuilder':
        """Clear all actions and reset the builder."""
        self._actions.clear()
        return self
