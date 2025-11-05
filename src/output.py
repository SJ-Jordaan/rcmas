"""Output and file management for the RCMAS system."""
from .solver import find_solution, block_solution
from .config import GRID_HEIGHT, GRID_WIDTH, NUM_AGENTS, NUM_TIMESTEPS
from .visualization import get_grid_string, display_grid


def write_model_to_file(file_handle, model, model_count):
  """
  Write a single model to the output file.

  Args:
      file_handle: Open file handle to write to
      model: Z3 model to write
      model_count: Current model number
  """
  file_handle.write(f"{'=' * 60}\n")
  file_handle.write(f"Model {model_count}\n")
  file_handle.write(f"{'=' * 60}\n\n")
  file_handle.write(f"Raw model:\n{model}\n\n")
  file_handle.write(get_grid_string(model, GRID_HEIGHT, GRID_WIDTH, NUM_TIMESTEPS, NUM_AGENTS))
  file_handle.write("\n\n")


def save_all_models(solver, output_filename="models_output.txt", max_models=1):
  """
  Find and save all models to a file.

  Args:
      solver: Configured Z3 solver
      output_filename: Name of the output file
      max_models: Maximum number of models to find (default: 1)

  Returns:
      int: Number of models found
  """
  model_count = 0

  with open(output_filename, "w") as f:
    while model_count < max_models:
      model = find_solution(solver)

      if model is None:
        break

      model_count += 1

      write_model_to_file(f, model, model_count)

      print(f"Model {model_count} found and written to file")
      display_grid(model, GRID_HEIGHT, GRID_WIDTH, NUM_TIMESTEPS, NUM_AGENTS)

      block_solution(solver, model)

    if model_count == 0:
      print("No solution exists.")
      f.write("No solution exists.\n")
    else:
      print(f"\nTotal models found: {model_count}")
      f.write(f"Total models found: {model_count}\n")

  return model_count
