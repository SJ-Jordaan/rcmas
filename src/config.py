"""Configuration constants for the RCMAS system."""

# Grid dimensions
GRID_HEIGHT = 2
GRID_WIDTH = 2
NEUTRAL_SECTORS = []

# Agent configuration
NUM_AGENTS = 2

# Derived values
NUM_SECTORS = GRID_HEIGHT * GRID_WIDTH
NUM_TIMESTEPS = NUM_SECTORS // NUM_AGENTS
