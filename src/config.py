"""Configuration constants for the RCMAS system."""

# Grid dimensions
GRID_HEIGHT = 3
GRID_WIDTH = 3

# Agent configuration
NUM_AGENTS = 3

# Derived values
NUM_SECTORS = GRID_HEIGHT * GRID_WIDTH
NUM_TIMESTEPS = NUM_SECTORS // NUM_AGENTS
