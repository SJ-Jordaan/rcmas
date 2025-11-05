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


def for_all_timesteps(func):
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


def for_all_sectors(func):
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


def for_all_agents(func):
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


def for_all_timesteps_and_sectors(func):
  """
  Apply a function for all timestep-sector combinations.

  Args:
      func: Function that takes (t, sector_id) and returns constraints

  Returns:
      list: Flattened list of all constraints
  """
  return for_all_timesteps(
    lambda t: for_all_sectors(
      lambda sector_id: func(
        t, sector_id
      )
    )
  )


def for_all_timesteps_and_agents(func):
  """
  Apply a function for all timestep-agent combinations.

  Args:
      func: Function that takes (t, agt) and returns constraints

  Returns:
      list: Flattened list of all constraints
  """
  return for_all_timesteps(
    lambda t: for_all_agents(
      lambda agt: func(
        t, agt
      )
    )
  )


def for_all_timesteps_sectors_and_agents(func):
  """
  Apply a function for all timestep-sector-agent combinations.

  Args:
      func: Function that takes (t, sector_id, agt) and returns constraints

  Returns:
      list: Flattened list of all constraints
  """
  return for_all_timesteps(
    lambda t: for_all_sectors(
      lambda sector_id: for_all_agents(
        lambda agt: func(
          t, sector_id, agt
        )
      )
    )
  )


def for_all_agent_pairs(func):
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


def adjacency_pairs(grid_height, grid_width):
  """
  Generate all adjacent sector pairs using 4-neighborhood (right and down only).

  Args:
      grid_height: Height of the grid
      grid_width: Width of the grid

  Returns:
      list: List of tuples (u, v) representing adjacent sector pairs
  """
  pairs = []
  for r in range(grid_height):
    for c in range(grid_width):
      u = r * grid_width + c
      # Right neighbor
      if c + 1 < grid_width:
        v = r * grid_width + (c + 1)
        pairs.append((u, v))
      # Down neighbor
      if r + 1 < grid_height:
        v = (r + 1) * grid_width + c
        pairs.append((u, v))
  return pairs
