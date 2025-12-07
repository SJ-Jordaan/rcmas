"""Visualization and display functions for the RCMAS system."""
from z3 import Bool
from utils.coords import calculate_position

def format_grid(model, state, grid_height, grid_width, timestep, inaccessible_sectors):
  """
      Format a grid with a 'box' layout for each cell.
      Shows Sector ID, Coordinates (row, col), and the Agent/Value.
      """
  rows_output = []

  # Calculate dynamic width based on grid size to ensure text fits
  # Example fit: "S:99 (9,9)" -> needs ~10-12 chars
  cell_width = 12

  # Create the horizontal divider line (e.g., +------------+------------+)
  divider = "+" + ("-" * cell_width + "+") * grid_width

  rows_output.append(divider)

  for row in range(grid_height):
    # We need two lines of text per grid row:
    # 1. The Info Line (Sector ID and Coords)
    # 2. The Value Line (The 'X', '.', or Agent ID)
    line_info = "|"
    line_value = "|"

    for col in range(grid_width):
      sector_id = row * grid_width + col

      # --- 1. Logic to determine the value (unchanged) ---
      if sector_id in inaccessible_sectors:
        val_str = "X"
      else:
        sector_var = state[sector_id][timestep]
        try:
          value = model.eval(sector_var, model_completion=True).as_long()
          if value == 0:
            val_str = "."
          else:
            val_str = str(value)
        except Exception:
          val_str = "?"

      # --- 2. Formatting the cell internals ---

      # Top part: Sector and Coords (e.g., "S:5 (1,2)")
      # We truncate or pack it to fit cell_width
      info_text = f"S:{sector_id} ({row},{col})"

      # Bottom part: The actual value centered
      # If it's a dot, we keep it subtle; if it's an agent/X, it stands out
      val_text = val_str

      # Append to the current string builders with padding
      # ^ centers the text, < left aligns
      line_info += f" {info_text:^{cell_width - 2}} |"
      line_value += f" {val_text:^{cell_width - 2}} |"

    # Add the constructed lines to the output
    rows_output.append(line_info)
    rows_output.append(line_value)
    rows_output.append(divider)

  return rows_output


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


def format_cohesion_grid(model, state, grid_height, grid_width, num_timesteps, num_agents, num_sectors):
  """
    Displays raw variable data without BFS interpretation:
    1. The Connectivity Matrix (Adjacency Matrix) for the final `cr` variables.
    2. The Size Map (The value of the `size` variable per sector).
    """
  output = ["=" * 60]
  output.append("RAW SOLVER DATA (No post-processing)")
  output.append("=" * 60)

  # --- 1. Connectivity Matrix (Visualizing the cr variables) ---
  # This shows exactly which sectors the solver thinks are connected.
  for a in range(num_agents):
    agent_id = a + 1  # 1-based index for display
    output.append(f"\n[Agent {agent_id} Connectivity Matrix]")
    output.append(f"('X' = cr(i,j) is True, '.' = False)")

    # Create Header (0 1 2 ...)
    header = "     " + " ".join([f"{i:<2}" for i in range(num_sectors)])
    output.append(header)
    output.append("    " + "-" * (len(header) - 4))

    for i in range(num_sectors):
      row_str = f"{i:<2} | "
      for j in range(num_sectors):
        if i == j:
          row_str += "\\  "  # Self
          continue

        # Sort indices because variables are usually stored as i_j with i < j
        u, v = (i, j) if i < j else (j, i)

        val_char = "."
        try:
          # Attempt to read the specific boolean variable
          # Adjust the variable name format to match your encoding exactly
          cr_name = f"cohesive_relation_{u}_{v}_{a}"
          cr_var = Bool(cr_name)

          # Check if True in the model
          if model.eval(cr_var, model_completion=True):
            val_char = "X"
        except Exception:
          val_char = "?"

        row_str += f"{val_char:<3}"
      output.append(row_str)

  # --- 2. Size Variable Map ---
  # Displays the integer value of 'size_{s}_{a}' on the grid
  output.append("\n" + "=" * 60)
  output.append("Cluster Size Map (Value of 'size' variable)")
  output.append("=" * 60)

  for a in range(num_agents):
    agent_id = a + 1
    output.append(f"\n[Agent {agent_id} Calculated Sizes]")

    # Draw grid for this agent
    rows_output = []
    # Simple divider
    divider = "+" + ("-" * 5 + "+") * grid_width
    rows_output.append(divider)

    for row in range(grid_height):
      line_val = "|"
      for col in range(grid_width):
        sector_id = row * grid_width + col

        # Retrieve the 'size' variable directly
        # Assumes variable is named 'size_{sector}_{agent}' (0-indexed agent)
        val_str = " "
        try:
          # Variable name from your previous code: size(s, a) -> size_{s}_{a}
          size_var_name = f"size_{sector_id}_{a}"
          # We need to access the integer variable.
          # Since we don't have the variable object, we create a reference with the same name
          from z3 import Int
          s_var = Int(size_var_name)

          val = model.eval(s_var, model_completion=True).as_long()
          if val > 0:
            val_str = str(val)
          else:
            val_str = "."
        except Exception:
          val_str = "?"

        line_val += f" {val_str:^3} |"

      rows_output.append(line_val)
      rows_output.append(divider)

    output.extend(rows_output)

  return output


def display_grid(model, state, action, grid_height, grid_width, num_timesteps, num_agents, inaccessible_sectors):
  """
  Display the grid for all timesteps to the console.
  """
  print(get_grid_string(model, state, action, grid_height, grid_width, num_timesteps, num_agents, inaccessible_sectors))


def get_grid_string(model, state, action, grid_height, grid_width, num_timesteps, num_agents, inaccessible_sectors):
  """
  Get the grid display as a string for all timesteps.
  """
  output = []
  for t in range(num_timesteps + 1):
    output.append(f"Timestep {t}:")

    # Pass 'state'
    grid_rows = format_grid(model, state, grid_height, grid_width, t, inaccessible_sectors)
    output.extend(grid_rows)

    if t < num_timesteps:
      output.append(f"Actions at t={t}:")
      # Pass 'action'
      actions = format_actions(model, action, grid_width, num_agents, t)
      output.extend(actions)

    output.append("")

  # --- NEW VISUALIZATION ---
  # Add the cohesion grid to the output
  num_sectors = grid_height * grid_width
  cohesion_rows = format_cohesion_grid(
    model, state, grid_height, grid_width, num_timesteps, num_agents, num_sectors
  )
  output.extend(cohesion_rows)
  # --- END NEW VISUALIZATION ---

  return "\n".join(output)
