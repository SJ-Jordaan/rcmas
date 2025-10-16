"""
Agent class representing autonomous agents in the RCMAS.
"""

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from strategies.abstract_strategy import AbstractStrategy


@dataclass
class Agent:
    """
    Represents an agent a_k ∈ Agt in the RCMAS.
    
    Each agent has:
    - A unique identifier
    - An objective o(a) defining the minimum cohesive region size to control
    - A strategy for selecting actions
    """
    
    id: str
    objective: int
    strategy: Optional['AbstractStrategy'] = None
    
    def __post_init__(self):
        """Validate agent configuration."""
        if self.objective < 0:
            raise ValueError(f"Agent objective must be non-negative, got {self.objective}")
        
        if not self.id:
            raise ValueError("Agent must have a non-empty id")
    
    def __str__(self) -> str:
        return f"Agent({self.id}, obj={self.objective})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other: object) -> bool:
        """Agents are equal if they have the same id."""
        if not isinstance(other, Agent):
            return NotImplemented
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on agent id for use in sets and dicts."""
        return hash(self.id)
    
    def set_strategy(self, strategy: 'AbstractStrategy') -> None:
        """
        Set or update the agent's strategy.
        
        Args:
            strategy: The strategy to use for action selection
        """
        self.strategy = strategy
    
    def has_strategy(self) -> bool:
        """Check if the agent has a strategy assigned."""
        return self.strategy is not None
    
    def get_objective(self) -> int:
        """Returns the minimum region size objective for this agent."""
        return self.objective


class DummyAgent(Agent):
    """
    Special agent a_0 representing unoccupied sectors.
    
    This is a singleton instance used to indicate that a sector
    is currently not controlled by any real agent.
    """
    
    _instance: Optional['DummyAgent'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            super().__init__(id="a0", objective=0, strategy=None)
            self._initialized = True
    
    def __str__(self) -> str:
        return "DummyAgent(a0)"
    
    def __repr__(self) -> str:
        return "DummyAgent()"


# Singleton instance for unoccupied sectors
DUMMY_AGENT = DummyAgent()


class AgentFactory:
    """
    Factory for creating agents with common configurations.
    """
    
    _agent_counter = 0
    
    @classmethod
    def create_agent(
        cls,
        objective: int,
        strategy: Optional['AbstractStrategy'] = None,
        id_prefix: str = "a"
    ) -> Agent:
        """
        Create a new agent with an auto-generated id.
        
        Args:
            objective: Minimum region size objective
            strategy: Optional strategy for the agent
            id_prefix: Prefix for the auto-generated id
            
        Returns:
            A new Agent instance
        """
        cls._agent_counter += 1
        agent_id = f"{id_prefix}{cls._agent_counter}"
        return Agent(id=agent_id, objective=objective, strategy=strategy)
    
    @classmethod
    def create_agents(
        cls,
        count: int,
        objective: int,
        strategy_factory: Optional[callable] = None,
        id_prefix: str = "a"
    ) -> list[Agent]:
        """
        Create multiple agents with the same objective.
        
        Args:
            count: Number of agents to create
            objective: Objective for all agents
            strategy_factory: Optional callable that returns a new strategy instance
            id_prefix: Prefix for agent ids
            
        Returns:
            List of Agent instances
        """
        agents = []
        for _ in range(count):
            strategy = strategy_factory() if strategy_factory else None
            agent = cls.create_agent(objective, strategy, id_prefix)
            agents.append(agent)
        return agents
    
    @classmethod
    def reset_counter(cls) -> None:
        """Reset the agent counter (useful for testing)."""
        cls._agent_counter = 0
