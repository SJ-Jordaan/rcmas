# src/handlers/solver.py
from z3 import sat
from src.handlers.base import Handler
from src.core.state import PipelineContext
from src.logic.constraints import encode_block_solution


class Z3SolverHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    ctx.found_models = []  # Initialize list to store found models

    # We assume encoder step has already populated variables in context
    # You might need to store state/action vars in context during Encoding step to access them here
    state_vars = ctx.z3_vars['state']
    action_vars = ctx.z3_vars['action']

    max_models = ctx.config.simulation.max_models  # Define this in your config/pydantic
    model_count = 0

    print(f"Starting solver loop (Target: {max_models} models)...")

    while model_count < max_models:
      check_result = ctx.z3_optimizer.check()

      if check_result == sat:
        model = ctx.z3_optimizer.model()
        ctx.found_models.append(model)
        model_count += 1
        ctx.is_satisfiable = True
        print(f"  -> Model {model_count} found.")

        # If we need more models, block this one
        if model_count < max_models:
          block_constraint = encode_block_solution(
            model,
            state_vars,
            action_vars,
            ctx.num_sectors,
            ctx.num_timesteps,
            ctx.config.agents.count
          )
          ctx.z3_optimizer.add(block_constraint)
      else:
        print("  -> No further models found.")
        break

    if model_count == 0:
      ctx.is_satisfiable = False

    return ctx
