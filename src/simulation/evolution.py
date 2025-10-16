"""
Evolution relation implementing state transitions in RCMAS.
"""

from typing import Optional
from domain.state import State, FailureState
from domain.action import ActionProfile
from domain.agent import DUMMY_AGENT


class Evolution:
    """
    Implements the evolution relation δ ⊆ S × AP × S.
    
    Defines how the system transitions from one state to another
    based on the execution of an action profile.
    """
    
    @staticmethod
    def apply(state: State, action_profile: ActionProfile) -> State:
        """
        Apply an action profile to a state to produce a successor state.
        
        Implements the evolution rules from the paper:
        1. If s(i,j) = a_0 then:
           a. If no agent targets (i,j), then s'(i,j) = a_0
           b. If exactly one agent targets (i,j), then s'(i,j) = that agent
           c. If multiple agents target (i,j), then s' = s^x (failure)
        2. If s(i,j) = a for some agent a, then s'(i,j) = a (persistence)
        
        Args:
            state: The current state
            action_profile: The joint action chosen by all agents
            
        Returns:
            The successor state (may be FailureState if collision occurs)
        """
        # Check for collisions first
        if action_profile.has_collision():
            # Return failure state that preserves current occupancy
            failure_state = FailureState(state.get_territory())
            # Copy the current occupancy into the failure state
            for sector in state.get_territory().sectors:
                occupant = state.get_occupant(sector)
                failure_state.set_occupant(sector, occupant)
            return failure_state
        
        # Create new state as a copy of current state
        new_state = state.copy()
        
        # Get sector targets
        sector_targets = action_profile.get_target_sectors()
        
        # Apply each action
        for sector, agents in sector_targets.items():
            # This should always be exactly one agent due to collision check above
            assert len(agents) == 1, "Collision should have been detected"
            agent = next(iter(agents))
            
            # Only update if the sector is currently unoccupied
            if state.is_unoccupied(sector):
                new_state.set_occupant(sector, agent)
        
        return new_state
    
    @staticmethod
    def is_terminal(state: State) -> bool:
        """
        Check if a state is terminal (simulation should end).
        
        A state is terminal if:
        - It is a failure state (s^x), OR
        - All sectors are occupied
        
        Args:
            state: The state to check
            
        Returns:
            True if terminal, False otherwise
        """
        return state.is_failure() or state.is_fully_occupied()
    
    @staticmethod
    def validate_action_profile(state: State, action_profile: ActionProfile) -> bool:
        """
        Validate that an action profile is executable in a state.
        
        An action profile is executable if all actions target unoccupied sectors.
        
        Args:
            state: The current state
            action_profile: The action profile to validate
            
        Returns:
            True if executable, False otherwise
        """
        for agent, action in action_profile.get_all_actions().items():
            # Check that the action's agent matches the profile's agent
            if action.agent != agent:
                return False
            
            # Check that the target sector is unoccupied
            if not state.is_unoccupied(action.sector):
                return False
        
        return True
