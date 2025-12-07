# src/handlers/output.py
from src.handlers.base import Handler
from src.core.state import PipelineContext
from src.logic.visualization import get_grid_string


class OutputHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    if not ctx.found_models:
      print("No models to visualize.")
      return ctx

    output_filename = "models_output.txt"

    # Retrieve logic variables from context
    state_vars = ctx.z3_vars['state']
    action_vars = ctx.z3_vars['action']

    with open(output_filename, "w") as f:
      f.write(f"Total models found: {len(ctx.found_models)}\n\n")

      for i, model in enumerate(ctx.found_models):
        model_num = i + 1
        f.write(f"{'=' * 60}\n")
        f.write(f"Model {model_num}\n")
        f.write(f"{'=' * 60}\n\n")

        # Use the restored visualization logic
        grid_str = get_grid_string(
          model,
          state_vars,
          action_vars,
          ctx.config.grid.height,
          ctx.config.grid.width,
          ctx.num_timesteps,
          ctx.config.agents.count,
          ctx.config.grid.inaccessible_sectors
        )
        f.write(grid_str)
        f.write("\n\n")

    print(f"Visualization written to {output_filename}")
    return ctx
