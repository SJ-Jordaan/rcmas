# src/handlers/solver.py
import logging
import time
from z3 import sat
from src.handlers.base import Handler
from src.core.state import PipelineContext
from src.logic.constraints import BlockSolutionConstraint


def _model_payoffs(ctx: PipelineContext, model):
  payoff_vars = ctx.z3_vars['payoff']
  per_agent = [model.eval(p, model_completion=True).as_long() for p in payoff_vars]
  return per_agent, sum(per_agent)


class SolverHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    logger = logging.getLogger("rcmas.solver")
    ctx.found_models = []
    max_models = ctx.config.simulation.max_models

    # Instantiate the blocker strategy once
    blocker = BlockSolutionConstraint()

    logger.info("Starting solver loop (target=%s)", max_models)
    start_time = time.perf_counter()

    while len(ctx.found_models) < max_models:
      loop_start = time.perf_counter()
      check_result = ctx.z3_optimizer.check()

      if check_result == sat:
        model = ctx.z3_optimizer.model()
        ctx.found_models.append(model)  # Push model to context
        ctx.z3_model = model
        ctx.is_satisfiable = True
        payoffs, payoff_sum = _model_payoffs(ctx, model)
        elapsed = time.perf_counter() - loop_start
        logger.info(
          "Model %s found | payoff_sum=%s | payoffs=%s | solve_time=%.3fs",
          len(ctx.found_models),
          payoff_sum,
          payoffs,
          elapsed,
        )

        # If we need to find more, block this one
        if len(ctx.found_models) < max_models:
          # The constraint class looks at ctx.found_models[-1] automatically
          blocking_clauses = blocker.build(ctx)
          ctx.z3_optimizer.add(blocking_clauses)
          logger.debug(
            "Blocking model %s with %s clauses",
            len(ctx.found_models),
            len(blocking_clauses),
          )
      else:
        logger.info("No further models found (sat=%s)", check_result)
        break

    total_time = time.perf_counter() - start_time
    logger.info(
      "Solver loop complete | models=%s | satisfiable=%s | elapsed=%.3fs",
      len(ctx.found_models),
      ctx.is_satisfiable,
      total_time,
    )

    if not ctx.is_satisfiable:
      if ctx.config.debug.include_strategy_constraints:
        logger.warning(
          "Unsat with strategy constraints. Set debug.include_strategy_constraints=false to test encoding only."
        )
      else:
        logger.warning("Unsat even without strategy constraints; check encoding/config.")

    return ctx
