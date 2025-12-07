from src.handlers.base import Handler
from src.core.state import PipelineContext


class LearningHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    if not ctx.is_satisfiable:
      # Reward punishment
      ctx.current_reward = -10
      # Mutate config to try and fix the problem (Example)
      ctx.config.agents.count = max(1, ctx.config.agents.count - 1)
      print(f"Solver failed. Reducing agents to {ctx.config.agents.count}")
    else:
      # Calculate reward based on model output
      ctx.current_reward = 10
      # Update Q-Table (Placeholder)
      pass

    ctx.iteration += 1
    return ctx
