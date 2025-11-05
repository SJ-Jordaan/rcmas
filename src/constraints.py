"""Constraint generation for the RCMAS system."""

from z3 import Int, Implies, And, Or
from .config import NEUTRAL_SECTORS, NUM_SECTORS, NUM_AGENTS, NUM_TIMESTEPS
from .utils import (
    forAllTimestepsSectorsAndAgents,
    forAllTimestepsAndAgents,
)

# Hard constraints

def generate_action_availability_constraints():
    """
    Generate constraints ensuring agents can only act on available sectors.
    Also enforces action bounds (0 <= action < NUM_SECTORS).
    
    Returns:
        list: List of action availability constraints
    """
    # If a sector is occupied or neutral, no agent can choose it
    occupied_sector_constraints = forAllTimestepsSectorsAndAgents(
        lambda t, sector_id, agt: Implies(
            Or(
                Int(f"sector_{sector_id}_{t}") != 0,
                sector_id in NEUTRAL_SECTORS
            ),
            Int(f"action_{agt}_{t}") != sector_id
        )
    )
    
    # Action lower bounds (>= 0)
    lower_bound_constraints = forAllTimestepsAndAgents(
        lambda t, agt: 0 <= Int(f"action_{agt}_{t}")
    )
    
    # Action upper bounds (< NUM_SECTORS)
    upper_bound_constraints = forAllTimestepsAndAgents(
        lambda t, agt: Int(f"action_{agt}_{t}") < NUM_SECTORS
    )
    
    return [
        *occupied_sector_constraints,
        *lower_bound_constraints,
        *upper_bound_constraints
    ]


def generate_evolution_constraints():
    """
    Generate constraints for state evolution when agents act successfully.
    
    Returns:
        list: List of evolution constraints
    """
    # If an agent chooses a sector and no other agent chooses it, that agent occupies it
    successful_action_constraints = forAllTimestepsSectorsAndAgents(
        lambda t, sector_id, agt: Implies(
            And(
                Int(f"action_{agt}_{t}") == sector_id,
                *[Int(f"action_{other_agent}_{t}") != sector_id
                  for other_agent in range(1, NUM_AGENTS + 1)
                  if other_agent != agt]
            ),
            Int(f"sector_{sector_id}_{t+1}") == agt
        )
    )
    
    # If no agent chooses a sector, it remains unchanged
    from .utils import forAllTimestepsAndSectors, forAllAgents
    unchanged_sector_constraints = forAllTimestepsAndSectors(
        lambda t, sector_id: Implies(
            And(forAllAgents(lambda agt: Int(f"action_{agt}_{t}") != sector_id)),
            Int(f"sector_{sector_id}_{t}") == Int(f"sector_{sector_id}_{t+1}")
        )
    )
    
    return [
        *successful_action_constraints,
        *unchanged_sector_constraints
    ]


def generate_conflict_constraints():
    """
    Generate constraints for conflict resolution when multiple agents choose the same sector.
    
    Returns:
        list: List of conflict resolution constraints
    """
    from .utils import forAllTimestepsAndSectors, forAllAgentPairs
    
    # If two agents choose the same sector, it remains unchanged (conflict)
    return forAllTimestepsAndSectors(
        lambda t, sector_id: forAllAgentPairs(
            lambda agt1, agt2: Implies(
                And(
                    Int(f"action_{agt1}_{t}") == sector_id,
                    Int(f"action_{agt2}_{t}") == sector_id,
                ),
                Int(f"sector_{sector_id}_{t}") == Int(f"sector_{sector_id}_{t+1}")
            )
        )
    )


def generate_full_board_constraints():
    """
    Generate constraints ensuring the board is fully occupied at the final timestep.
    Excludes neutral sectors which cannot be occupied.
    
    Returns:
        list: List of full board constraints
    """
    return [
        Int(f"sector_{sector_id}_{NUM_TIMESTEPS}") != 0
        for sector_id in range(NUM_SECTORS)
        if sector_id not in NEUTRAL_SECTORS
    ]


def generate_all_hard_constraints():
    """
    Generate all constraints for the RCMAS system.
    
    Returns:
        list: List of all constraints
    """
    return (
        generate_action_availability_constraints() +
        generate_evolution_constraints() +
        generate_conflict_constraints() +
        generate_full_board_constraints()
    )

# Soft constraints

        

def generate_maximise_cohesive_region_constraints(timestep=None):
    """
    Generate soft constraints to maximise the size of cohesive regions occupied by agents.
    A cohesive region is defined as a group of adjacent sectors occupied by the same agent.
    
    Uses adjacency pairs (4-neighborhood) as a proxy for connected regions. Maximizes
    the number of adjacent sector pairs owned by the same agent at a given timestep.
    Excludes pairs involving neutral sectors.
    
    Args:
        timestep: Timestep to evaluate (default: NUM_TIMESTEPS, i.e., final state)
    
    Returns:
        list: List of Bool expressions for soft constraints (adjacency-owned-by-same-agent)
    """
    from z3 import And
    from .config import GRID_HEIGHT, GRID_WIDTH
    from .utils import adjacency_pairs
    
    if timestep is None:
        timestep = NUM_TIMESTEPS
    
    soft_constraints = []
    for (u, v) in adjacency_pairs(GRID_HEIGHT, GRID_WIDTH):
        # Skip pairs involving neutral sectors
        if u in NEUTRAL_SECTORS or v in NEUTRAL_SECTORS:
            continue
            
        for agt in range(1, NUM_AGENTS + 1):
            # Reward when both adjacent sectors are owned by the same agent
            soft_constraints.append(And(
                Int(f"sector_{u}_{timestep}") == agt,
                Int(f"sector_{v}_{timestep}") == agt,
            ))
    
    return soft_constraints
    