"""Utility functions for the RCMAS system."""

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

def sector_to_coords(sector, grid_width):
  row = sector // grid_width
  col = sector % grid_width
  return row, col
