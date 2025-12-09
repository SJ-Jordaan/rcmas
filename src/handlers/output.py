# src/handlers/output.py
import logging
from datetime import datetime
from pathlib import Path
from src.handlers.base import Handler
from src.core.state import PipelineContext
from src.logic.visualization import get_grid_string
from src.utils.debug import top_actions, format_changes_table


class OutputHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    logger = logging.getLogger("rcmas.output")
    if not ctx.found_models:
      logger.warning("No models to visualize.")
      return ctx

    def _compute_payoffs(model):
      payoff_vars = ctx.z3_vars['payoff']
      per_agent = [model.eval(p, model_completion=True).as_long() for p in payoff_vars]
      return per_agent, sum(per_agent)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = ctx.target_agent_idx if ctx.target_agent_idx is not None else "all"
    output_filename = f"models_output_{ctx.mode.value}_target_{target}_{run_tag}.txt"
    path = out_dir / output_filename

    # Retrieve logic variables from context
    state_vars = ctx.z3_vars['state']
    action_vars = ctx.z3_vars['action']

    with open(path, "w") as f:
      # Header
      f.write("RCMAS Solver Output\n")
      f.write(f"Grid: {ctx.config.grid.height}x{ctx.config.grid.width}\n")
      f.write(f"Agents: {ctx.config.agents.count}\n")
      f.write(f"Inaccessible: {ctx.config.grid.inaccessible_sectors}\n")
      f.write(f"Timesteps: {ctx.num_timesteps}\n")
      f.write(f"Models: {len(ctx.found_models)}\n")
      f.write(f"Last payoff: {ctx.last_payoff}\n\n")

      if ctx.last_path:
        f.write("Last path (state -> action)\n")
        max_state_logs = ctx.config.debug.max_state_logs
        for idx, (state_key, joint_action) in enumerate(ctx.last_path[:max_state_logs]):
          f.write(f"  t={idx}: state={state_key} action={joint_action}\n")
        if len(ctx.last_path) > max_state_logs:
          f.write(f"  ... truncated ({max_state_logs} of {len(ctx.last_path)})\n")
        f.write("\n")

      if ctx.last_q_changes:
        f.write("Q-value changes (capped)\n")
        f.write(format_changes_table(ctx.last_q_changes, max_rows=ctx.config.debug.max_q_deltas))
        f.write("\n\n")

      top = top_actions(ctx.q_table, limit=10)
      if top:
        f.write("Top Q actions (state -> best_action -> value)\n")
        for value, state_key, joint_action in top:
          f.write(f"  {state_key} -> {joint_action} -> {value:.3f}\n")
        f.write("\n")

      f.write(f"Total models found: {len(ctx.found_models)}\n\n")

      for i, model in enumerate(ctx.found_models):
        model_num = i + 1
        f.write(f"{'=' * 60}\n")
        f.write(f"Model {model_num}\n")
        f.write(f"{'=' * 60}\n\n")

        payoffs, payoff_sum = _compute_payoffs(model)
        f.write(f"Payoffs (per agent): {payoffs} | payoff_sum={payoff_sum}\n\n")

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

      logger.info("Visualization written to %s", path)
    return ctx
