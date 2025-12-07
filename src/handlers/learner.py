from logic.initialiser import ValidStateInitializer
from handlers.base import Handler
from core.state import PipelineContext, PipelineMode

def _extract_path(ctx):
  """Helper to get [(State, JointAction)] trace from Z3 model"""
  model = ctx.z3_model
  state_vars = ctx.z3_vars['state']
  action_vars = ctx.z3_vars['action']
  path = []

  for t in range(ctx.num_timesteps):
    # Get State
    s_vals = tuple(model.eval(state_vars[s][t], model_completion=True).as_long()
                   for s in range(ctx.num_sectors))

    # Get Joint Action
    a_vals = tuple(model.eval(action_vars[a][t], model_completion=True).as_long()
                   for a in range(ctx.config.agents.count))

    path.append((s_vals, a_vals))
  return path


def _get_total_payoff(ctx):
  model = ctx.z3_model
  payoff_vars = ctx.z3_vars['payoff']
  return sum(model.eval(p, model_completion=True).as_long() for p in payoff_vars)


class LearningHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    # 1. INITIALIZATION (Run once)
    if not ctx.q_table:
      print("[Learning] No Q-Table found. Initializing...")

      # Use the new Initializer Class
      initializer = ValidStateInitializer()
      ctx.q_table = initializer.generate(ctx)

      ctx.mode = PipelineMode.EVAL_BASELINE
      return ctx

    # 2. UPDATE STEP
    if ctx.is_satisfiable and ctx.z3_model:
      payoff = _get_total_payoff(ctx)

      # Extract the path taken in this run
      path = _extract_path(ctx)

      # Update Q-Table for visited states
      for state_key, joint_action in path:
        # Basic update: Set Q-value to the payoff found
        # (In real RL, use learning rate: Q = (1-a)Q + a*Reward)
        if state_key in ctx.q_table:
          ctx.q_table[state_key][joint_action] = float(payoff)

      # 3. LOGIC CONTROL (Switch Modes)
      if ctx.mode == PipelineMode.EVAL_BASELINE:
        print(f"[Learning] Baseline Payoff: {payoff}")
        ctx.current_baseline_payoff = payoff
        ctx.mode = PipelineMode.AGENT_OPTIMIZATION
        ctx.target_agent_idx = 0

      elif ctx.mode == PipelineMode.AGENT_OPTIMIZATION:
        if payoff > ctx.current_baseline_payoff:
          print(f"   -> Agent {ctx.target_agent_idx} improved payoff to {payoff}")
          ctx.current_baseline_payoff = payoff

        # Next agent
        next_agent = ctx.target_agent_idx + 1
        if next_agent < ctx.config.agents.count:
          ctx.target_agent_idx = next_agent
        else:
          print("[Learning] Round complete. Starting new episode.")
          ctx.iteration += 1
          ctx.mode = PipelineMode.EVAL_BASELINE
          ctx.target_agent_idx = None
    else:
      print("[Learning] Solver failed. Reverting to baseline.")
      ctx.mode = PipelineMode.EVAL_BASELINE
      ctx.target_agent_idx = None

    return ctx
