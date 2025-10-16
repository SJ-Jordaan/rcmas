"""
State class representing the state of an RCMAS.
"""

from typing import Dict, Set, Optional, FrozenSet
from collections import defaultdict

from domain.agent import Agent, DUMMY_AGENT
from domain.territory import Territory, Sector


class State:
    """
    Represents a state s: T → Agt⁺ of an RCMAS.
    
    A state maps each sector to the agent currently occupying it,
    or to the dummy agent a_0 if unoccupied.
    """
    
    def __init__(self, territory: Territory, occupancy: Optional[Dict[Sector, Agent]] = None):
        """
        Initialize a state.
        
        Args:
            territory: The territory this state is defined over
            occupancy: Optional mapping of sectors to agents (defaults to all unoccupied)
        """
        self._territory = territory
        self._occupancy: Dict[Sector, Agent] = {}
        
        if occupancy is None:
            # Initialize with all sectors unoccupied
            for sector in territory.sectors:
                self._occupancy[sector] = DUMMY_AGENT
        else:
            # Validate and copy provided occupancy
            for sector in territory.sectors:
                if sector in occupancy:
                    self._occupancy[sector] = occupancy[sector]
                else:
                    self._occupancy[sector] = DUMMY_AGENT
    
    @classmethod
    def create_initial_state(cls, territory: Territory) -> 'State':
        """
        Create the initial state s_0 where all sectors are unoccupied.
        
        Args:
            territory: The territory for the state
            
        Returns:
            Initial state with all sectors unoccupied
        """
        return cls(territory)
    
    def get_occupant(self, sector: Sector) -> Agent:
        """
        Get the agent occupying a sector.
        
        Args:
            sector: The sector to query
            
        Returns:
            The agent occupying the sector (DUMMY_AGENT if unoccupied)
            
        Raises:
            KeyError: If sector is not in the territory
        """
        if sector not in self._occupancy:
            raise KeyError(f"Sector {sector} not in territory")
        return self._occupancy[sector]
    
    def is_occupied(self, sector: Sector) -> bool:
        """
        Check if a sector is occupied by a real agent.
        
        Args:
            sector: The sector to check
            
        Returns:
            True if occupied by a real agent, False if unoccupied
        """
        return self.get_occupant(sector) != DUMMY_AGENT
    
    def is_unoccupied(self, sector: Sector) -> bool:
        """Check if a sector is unoccupied."""
        return not self.is_occupied(sector)
    
    def get_unoccupied_sectors(self) -> Set[Sector]:
        """
        Get all unoccupied sectors.
        
        Returns:
            Set of sectors occupied by DUMMY_AGENT
        """
        return {
            sector for sector, agent in self._occupancy.items()
            if agent == DUMMY_AGENT
        }
    
    def get_occupied_sectors(self) -> Set[Sector]:
        """
        Get all occupied sectors.
        
        Returns:
            Set of sectors occupied by real agents
        """
        return {
            sector for sector, agent in self._occupancy.items()
            if agent != DUMMY_AGENT
        }
    
    def get_sectors_for_agent(self, agent: Agent) -> Set[Sector]:
        """
        Get all sectors occupied by a specific agent.
        
        Args:
            agent: The agent to query
            
        Returns:
            Set of sectors controlled by the agent
        """
        return {
            sector for sector, occupant in self._occupancy.items()
            if occupant == agent
        }
    
    def get_agent_occupancy_counts(self) -> Dict[Agent, int]:
        """
        Get the count of sectors occupied by each agent.
        
        Returns:
            Dictionary mapping agents to their sector counts
        """
        counts: Dict[Agent, int] = defaultdict(int)
        for agent in self._occupancy.values():
            if agent != DUMMY_AGENT:
                counts[agent] += 1
        return dict(counts)
    
    def is_fully_occupied(self) -> bool:
        """
        Check if all sectors are occupied.
        
        Returns:
            True if no unoccupied sectors remain
        """
        return len(self.get_unoccupied_sectors()) == 0
    
    def get_territory(self) -> Territory:
        """Returns the territory this state is defined over."""
        return self._territory
    
    def copy(self) -> 'State':
        """Create a deep copy of this state."""
        return State(self._territory, self._occupancy.copy())
    
    def set_occupant(self, sector: Sector, agent: Agent) -> None:
        """
        Set the occupant of a sector (mutates this state).
        
        Args:
            sector: The sector to modify
            agent: The agent to set as occupant
            
        Raises:
            KeyError: If sector is not in territory
        """
        if sector not in self._occupancy:
            raise KeyError(f"Sector {sector} not in territory")
        self._occupancy[sector] = agent
    
    def __str__(self) -> str:
        occupied = len(self.get_occupied_sectors())
        total = len(self._territory)
        return f"State({occupied}/{total} occupied)"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return (self._territory == other._territory and 
                self._occupancy == other._occupancy)
    
    def __hash__(self) -> int:
        """
        Hash the state for use in sets/dicts.
        
        Note: This creates a frozen representation of occupancy.
        """
        frozen_occupancy = frozenset(self._occupancy.items())
        return hash((id(self._territory), frozen_occupancy))


class FailureState(State):
    """
    Special state s^x representing mission failure.
    
    This state is reached when a collision occurs (multiple agents
    attempting to occupy the same sector simultaneously).
    """
    
    _instance: Optional['FailureState'] = None
    
    def __new__(cls, territory: Territory):
        # Create a new instance each time since territory may differ
        # But we mark it as a failure state
        instance = object.__new__(cls)
        return instance
    
    def __init__(self, territory: Territory):
        super().__init__(territory)
        self._is_failure = True
    
    def is_failure(self) -> bool:
        """Returns True indicating this is a failure state."""
        return True
    
    def __str__(self) -> str:
        return "FailureState(mission failed due to collision)"
    
    def __repr__(self) -> str:
        return "FailureState()"


# Add failure checking method to base State class
State.is_failure = lambda self: isinstance(self, FailureState)
