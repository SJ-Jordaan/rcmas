"""Utility functions for the RCMAS system."""

from .config import GRID_HEIGHT, GRID_WIDTH

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

def sector_to_coords(sector):
  row = sector // GRID_WIDTH
  col = sector % GRID_WIDTH
  return row, col
