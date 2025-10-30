"""Utility functions for the RCMAS system."""

from .config import NUM_SECTORS, NUM_AGENTS, NUM_TIMESTEPS

def calculate_position(sector_id, grid_width):
    """
    Convert a sector ID to row and column coordinates.
    
    Args:
        sector_id: The sector identifier
        grid_width: Width of the grid
        
    Returns:
        tuple: (row, col) coordinates
    """
    row = sector_id // grid_width
    col = sector_id % grid_width
    return row, col


def generate_agent_pairs():
    """
    Generate all unique pairs of agents.
        
    Returns:
        list: List of tuples containing unique agent pairs
    """
    return [
        (agt1, agt2)
        for agt1 in range(1, NUM_AGENTS + 1)
        for agt2 in range(agt1 + 1, NUM_AGENTS + 1)
    ]


def forAllTimesteps(func):
    """
    Apply a function for all timesteps.
    
    Args:
        func: Function that takes timestep t and returns constraints
        
    Returns:
        list: Flattened list of all constraints
    """
    constraints = []
    for t in range(NUM_TIMESTEPS):
        result = func(t)
        if isinstance(result, list):
            constraints.extend(result)
        else:
            constraints.append(result)
    return constraints


def forAllSectors(func):
    """
    Apply a function for all sectors.
    
    Args:
        func: Function that takes sector_id and returns constraints
        
    Returns:
        list: Flattened list of all constraints
    """
    constraints = []
    for sector_id in range(NUM_SECTORS):
        result = func(sector_id)
        if isinstance(result, list):
            constraints.extend(result)
        else:
            constraints.append(result)
    return constraints


def forAllAgents(func):
    """
    Apply a function for all agents.
    
    Args:
        func: Function that takes agent id and returns constraints
        
    Returns:
        list: Flattened list of all constraints
    """
    constraints = []
    for agt in range(1, NUM_AGENTS + 1):
        result = func(agt)
        if isinstance(result, list):
            constraints.extend(result)
        else:
            constraints.append(result)
    return constraints


def forAllTimestepsAndSectors(func):
    """
    Apply a function for all timestep-sector combinations.
    
    Args:
        func: Function that takes (t, sector_id) and returns constraints
        
    Returns:
        list: Flattened list of all constraints
    """
    return forAllTimesteps(lambda t:
        forAllSectors(lambda sector_id:
            func(t, sector_id)
        )
    )


def forAllTimestepsAndAgents(func):
    """
    Apply a function for all timestep-agent combinations.
    
    Args:
        func: Function that takes (t, agt) and returns constraints
        
    Returns:
        list: Flattened list of all constraints
    """
    return forAllTimesteps(lambda t:
        forAllAgents(lambda agt:
            func(t, agt)
        )
    )


def forAllTimestepsSectorsAndAgents(func):
    """
    Apply a function for all timestep-sector-agent combinations.
    
    Args:
        func: Function that takes (t, sector_id, agt) and returns constraints
        
    Returns:
        list: Flattened list of all constraints
    """
    return forAllTimesteps(lambda t:
        forAllSectors(lambda sector_id:
            forAllAgents(lambda agt:
                func(t, sector_id, agt)
            )
        )
    )


def forAllAgentPairs(func):
    """
    Apply a function for all unique pairs of agents.
    
    Args:
        func: Function that takes (agt1, agt2) and returns constraints
        
    Returns:
        list: Flattened list of all constraints
    """
    constraints = []
    agent_pairs = generate_agent_pairs()
    for (agt1, agt2) in agent_pairs:
        result = func(agt1, agt2)
        if isinstance(result, list):
            constraints.extend(result)
        else:
            constraints.append(result)
    return constraints
