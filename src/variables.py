"""Z3 variable creation for the RCMAS system."""

from z3 import Int
from .config import NUM_SECTORS, NUM_AGENTS, NUM_TIMESTEPS


def create_sector_variables():
    """
    Create Z3 integer variables for all sectors across all timesteps.
    
    Returns:
        list: List of Z3 Int variables for sectors
    """
    return [
        Int(f"sector_{id}_{t}")
        for t in range(NUM_TIMESTEPS)
        for id in range(NUM_SECTORS)
    ]


def create_action_variables():
    """
    Create Z3 integer variables for agent actions across all timesteps.
    
    Returns:
        list: List of Z3 Int variables for actions
    """
    return [
        Int(f"action_{agt}_{t}")
        for t in range(NUM_TIMESTEPS)
        for agt in range(1, NUM_AGENTS + 1)
    ]


def create_initial_state():
    """
    Create constraints for the initial state (all sectors empty).
    
    Returns:
        list: List of constraints setting initial sector states to 0
    """
    return [Int(f"sector_{id}_0") == 0 for id in range(NUM_SECTORS)]
