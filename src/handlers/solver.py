# src/handlers/solver.py
from z3 import sat
from src.handlers.base import Handler
from src.core.state import PipelineContext
from src.logic.constraints import BlockSolutionConstraint


class SolverHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    ctx.found_models = []
    max_models = ctx.config.simulation.max_models

    # Instantiate the blocker strategy once
    blocker = BlockSolutionConstraint()

    print(f"Starting solver loop (Target: {max_models} models)...")

    while len(ctx.found_models) < max_models:
      check_result = ctx.z3_optimizer.check()

      if check_result == sat:
        model = ctx.z3_optimizer.model()
        ctx.found_models.append(model)  # Push model to context
        ctx.is_satisfiable = True
        print(f"  -> Model {len(ctx.found_models)} found.")

        # If we need to find more, block this one
        if len(ctx.found_models) < max_models:
          # The constraint class looks at ctx.found_models[-1] automatically
          blocking_clauses = blocker.build(ctx)
          ctx.z3_optimizer.add(blocking_clauses)
      else:
        print("  -> No further models found.")
        break

    return ctx
