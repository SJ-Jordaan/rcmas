"""Visualization and display functions for the RCMAS system."""
from z3 import Bool
# Import combinations in addition to permutations
from itertools import permutations, combinations
from .config import (
  INACCESSIBLE_SECTORS,
  NUM_AGENTS,
  NUM_TIMESTEPS,
  NUM_SECTORS,
  GRID_WIDTH,
)
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


def format_cohesion_grid(model, state, grid_height, grid_width, num_timesteps, num_agents, num_sectors):
  """
  Formats a grid showing the final cohesive regions for each agent.
  Sectors in the same region will have the same number.
  """
  output = ["=" * 60]
  output.append("Final Cohesion Grid (by Region ID)")
  output.append("=" * 60)

  for a in range(num_agents):
    agent_id = a + 1
    output.append(f"\nAgent {agent_id} Regions:")

    # 1. Build graph adjacency list from TRUE cohesive relations
    edges = {}
    nodes = set()

    # Find all sectors this agent occupies
    for i in range(num_sectors):
        try:
            if model.eval(state[i][num_timesteps], model_completion=True).as_long() == agent_id:
                nodes.add(i)
        except Exception:
            pass # Sector not in model or failed eval

    # --- BUG FIX ---
    # The cohesive relation is symmetric (i.e., cr_i_j == cr_j_i).
    # The previous loop used permutations(N, 2) and built a directed
    # graph, which caused the BFS to fail.
    #
    # The fix is to build an UNDIRECTED graph. We iterate using
    # combinations(N, 2) (which checks i < j) and add edges in
    # BOTH directions (i -> j and j -> i) if the relation is true.
    # This ensures the BFS will find the full connected component
    # regardless of the starting node.

    # Iterate using combinations (i < j)
    for i, j in combinations(range(num_sectors), 2):
      # Optimization: skip if neither node is owned by agent
      if i not in nodes and j not in nodes:
          continue

      is_cohesive = False
      try:
        # Check for relation i_j (assuming i < j)
        cr_var = Bool(f"cohesive_relation_{i}_{j}_{a}")
        is_cohesive = model.eval(cr_var, model_completion=True) == True
      except Exception:
        # If i_j fails, try j_i (defensive check)
        try:
          cr_var_sym = Bool(f"cohesive_relation_{j}_{i}_{a}")
          is_cohesive = model.eval(cr_var_sym, model_completion=True) == True
        except Exception:
          pass # Both failed, is_cohesive remains False

      # If we found a true relation, add the undirected edge
      if is_cohesive:
        # Add edge i -> j
        if i not in edges:
          edges[i] = []
        if j not in edges[i]:
          edges[i].append(j)

        # Add edge j -> i
        if j not in edges:
          edges[j] = []
        if i not in edges[j]:
          edges[j].append(i)
    # --- END BUG FIX ---

    # 2. Find connected components (regions) using BFS
    visited = set()
    regions = {}  # Map: sector_id -> region_id
    region_id_counter = 0

    # Iterate over all nodes owned by the agent
    for node in nodes:
      if node not in visited:
        region_id_counter += 1
        queue = [node]
        visited.add(node)

        while queue:
          current_node = queue.pop(0)
          regions[current_node] = region_id_counter

          if current_node in edges:
            for neighbor in edges[current_node]:
              # We only care about neighbors ALSO owned by this agent
              if neighbor in nodes and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # 3. Format the grid
    for row in range(grid_height):
      row_values = []
      for col in range(grid_width):
        sector_id = row * grid_width + col
        if sector_id in regions:
          # This sector is part of a found region
          val_str = str(regions[sector_id])
        elif sector_id in nodes:
          # This sector is occupied but was not in a cohesive relation
          # It's an isolated region of size 1.
          # Note: This logic might run if the BFS loop finishes before
          # processing all nodes (e.g., isolated nodes).
          region_id_counter += 1
          val_str = str(region_id_counter)
          regions[sector_id] = region_id_counter # Mark it
        else:
          # Not occupied by this agent
          val_str = "."
        row_values.append(f"{val_str:>2}")
      output.append(" ".join(row_values))

  return output


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

  # --- NEW VISUALIZATION ---
  # Add the cohesion grid to the output
  num_sectors = grid_height * grid_width
  cohesion_rows = format_cohesion_grid(
    model, state, grid_height, grid_width, num_timesteps, num_agents, num_sectors
  )
  output.extend(cohesion_rows)
  # --- END NEW VISUALIZATION ---

  return "\n".join(output)
