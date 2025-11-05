"""Visualization and display functions for the RCMAS system."""

from z3 import Int
from .config import NEUTRAL_SECTORS
from .utils import calculate_position


def format_grid(model, grid_height, grid_width, timestep):
    """
    Format a single timestep grid as a list of strings.
    Neutral sectors are displayed as 'X'.
    
    Args:
        model: Z3 model containing the solution
        grid_height: Height of the grid
        grid_width: Width of the grid
        timestep: The timestep to format
        
    Returns:
        list: List of strings representing grid rows
    """
    rows = []
    for row in range(grid_height):
        row_values = []
        for col in range(grid_width):
            sector_id = row * grid_width + col
            
            # Mark neutral sectors
            if sector_id in NEUTRAL_SECTORS:
                row_values.append("X")
            else:
                sector_var = Int(f"sector_{sector_id}_{timestep}")
                try:
                    value = model.eval(sector_var, model_completion=True)
                    row_values.append(str(value))
                except Exception:
                    row_values.append("?")
        rows.append(" ".join(row_values))
    return rows


def format_actions(model, grid_width, num_agents, timestep):
    """
    Format agent actions for a given timestep.
    
    Args:
        model: Z3 model containing the solution
        grid_width: Width of the grid
        num_agents: Total number of agents
        timestep: The timestep to format
        
    Returns:
        list: List of strings describing agent actions
    """
    actions = []
    for agt in range(1, num_agents + 1):
        action_var = Int(f"action_{agt}_{timestep}")
        try:
            action_value = model.eval(action_var, model_completion=True)
            sector_id = action_value.as_long()
            action_row, action_col = calculate_position(sector_id, grid_width)
            actions.append(
                f"  Agent {agt} -> sector {action_value} (row {action_row}, col {action_col})"
            )
        except Exception:
            actions.append(f"  Agent {agt} -> ?")
    return actions


def display_grid(model, grid_height, grid_width, num_timesteps, num_agents):
    """
    Display the grid for all timesteps to the console.
    
    Args:
        model: Z3 model containing the solution
        grid_height: Height of the grid
        grid_width: Width of the grid
        num_timesteps: Number of timesteps
        num_agents: Total number of agents
    """
    for t in range(num_timesteps + 1):
        print(f"Timestep {t}:")
        
        # Display grid
        grid_rows = format_grid(model, grid_height, grid_width, t)
        for row in grid_rows:
            print(row)
        
        # Display actions (if not the last timestep)
        if t < num_timesteps:
            print(f"Actions at t={t}:")
            actions = format_actions(model, grid_width, num_agents, t)
            for action in actions:
                print(action)
        
        print()


def get_grid_string(model, grid_height, grid_width, num_timesteps, num_agents):
    """
    Get the grid display as a string for all timesteps.
    
    Args:
        model: Z3 model containing the solution
        grid_height: Height of the grid
        grid_width: Width of the grid
        num_timesteps: Number of timesteps
        num_agents: Total number of agents
        
    Returns:
        str: Formatted string containing the entire grid display
    """
    output = []
    
    for t in range(num_timesteps + 1):
        output.append(f"Timestep {t}:")
        
        # Add grid
        grid_rows = format_grid(model, grid_height, grid_width, t)
        output.extend(grid_rows)
        
        # Add actions (if not the last timestep)
        if t < num_timesteps:
            output.append(f"Actions at t={t}:")
            actions = format_actions(model, grid_width, num_agents, t)
            output.extend(actions)
        
        output.append("")
    
    return "\n".join(output)
