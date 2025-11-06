"""Visualization and display functions for the RCMAS system."""
from .config import INACCESSIBLE_SECTORS
from .utils import calculate_position

def format_grid(model, state, grid_height, grid_width, timestep):
  """
    Format a single timestep grid with better alignment and characters.
    - '.' for empty (0)
    - 'X' for inaccessible
    - '1', '2', etc. for agents
    - All cells are right-aligned to 2 characters.
    """
  rows = []
  for row in range(grid_height):
    row_values = []
    for col in range(grid_width):
      sector_id = row * grid_width + col

      if sector_id in INACCESSIBLE_SECTORS:
        val_str = "X"
      else:
        sector_var = state[sector_id][timestep]
        try:
          value = model.eval(sector_var, model_completion=True).as_long()

          if value == 0:
            val_str = "."
          else:
            val_str = str(value)  # value will be 1, 2, -1, etc.
        except Exception:
          val_str = "?"  # Failed to evaluate

      # This is the key: f"{val_str:>2}"
      # It formats the string to be right-aligned (>) within 2 spaces.
      # '1' becomes ' 1', '10' becomes '10', '.' becomes ' .'
      row_values.append(f"{val_str:>2}")

    # Join with a single space for a clean grid
    rows.append(" ".join(row_values))
  return rows

def format_actions(model, action, grid_width, num_agents, timestep):
  """
  Format agent actions for a given timestep.

  Args:
      model: Z3 model containing the solution
      action: The nested list of action variables (action[a][t])
      grid_width: Width of the grid
      num_agents: Total number of agents
      timestep: The timestep to format
  """
  actions = []
  # Loop from 0 to NUM_AGENTS-1
  for a_idx in range(num_agents):
    # Agent ID for display is 1-based
    agt_display_id = a_idx + 1

    # Get the variable from our 0-indexed list
    action_var = action[a_idx][timestep]
    # --- END CHANGES ---

    try:
      action_value = model.eval(action_var, model_completion=True)
      sector_id = action_value.as_long()
      action_row, action_col = calculate_position(sector_id, grid_width)
      actions.append(
        f"  Agent {agt_display_id} -> sector {action_value} (row {action_row}, col {action_col})"
      )
    except Exception:
      actions.append(f"  Agent {agt_display_id} -> ?")
  return actions

def display_grid(model, state, action, grid_height, grid_width, num_timesteps, num_agents):
  """
  Display the grid for all timesteps to the console.
  """
  print(get_grid_string(model, state, action, grid_height, grid_width, num_timesteps, num_agents))


def get_grid_string(model, state, action, grid_height, grid_width, num_timesteps, num_agents):
  """
  Get the grid display as a string for all timesteps.
  """
  output = []
  for t in range(num_timesteps + 1):
    output.append(f"Timestep {t}:")

    # Pass 'state'
    grid_rows = format_grid(model, state, grid_height, grid_width, t)
    output.extend(grid_rows)

    if t < num_timesteps:
      output.append(f"Actions at t={t}:")
      # Pass 'action'
      actions = format_actions(model, action, grid_width, num_agents, t)
      output.extend(actions)

    output.append("")
  return "\n".join(output)
