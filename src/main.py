"""
RCMAS - Region Control Multi-Agent System

Main entry point for the system that coordinates all components.
"""

from src.config import GRID_HEIGHT, GRID_WIDTH, NUM_AGENTS, NUM_SECTORS, NUM_TIMESTEPS
from src.solver import setup_solver
from src.output import save_all_models


def main():
  """Main execution function."""
  print("=" * 60)
  print("RCMAS - Region Control Multi-Agent System")
  print("=" * 60)
  print(f"Grid: {GRID_HEIGHT}x{GRID_WIDTH}")
  print(f"Agents: {NUM_AGENTS}")
  print(f"Sectors: {NUM_SECTORS}")
  print(f"Timesteps: {NUM_TIMESTEPS}")
  print("=" * 60)
  print()

  solver = setup_solver()

  save_all_models(
    solver=solver,
    output_filename="models_output.txt",
    max_models=1  # Change this to find more models
  )


if __name__ == "__main__":
  main()
