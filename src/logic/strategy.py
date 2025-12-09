# src/logic/strategy.py
import logging
from pathlib import Path
from z3 import And, Implies
from logic.base import BaseConstraint
from core.state import PipelineContext, PipelineMode

class FixedStrategyConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    if not ctx.q_table:
      return []

    logger = logging.getLogger("rcmas.strategy")
    constraints = []
    state_vars = ctx.z3_vars['state']
    action_vars = ctx.z3_vars['action']

    fixed_agents = []
    if ctx.mode == PipelineMode.EVAL_BASELINE:
      fixed_agents = list(range(ctx.config.agents.count))
    elif ctx.mode == PipelineMode.AGENT_OPTIMIZATION:
      fixed_agents = [a for a in range(ctx.config.agents.count) if a != ctx.target_agent_idx]

    summary = []

    for state_tuple, action_map in ctx.q_table.items():

      # Find the BEST Joint Action for this state
      # action_map is Dict[JointActionTuple, Value]
      # We want the Key with the Max Value
      if not action_map:
        continue

      best_joint_action = max(action_map, key=action_map.get)

      # Add constraints for every timestep
      for t in range(ctx.num_timesteps):

        # This is the condition, basically identifying if we have found the state
        is_state_match = And([
          state_vars[s][t] == val
          for s, val in enumerate(state_tuple)
        ])

        # which actions to take when we have fixed an agent for that state
        implications = []
        for agent_id in fixed_agents:
          required_action = best_joint_action[agent_id]
          implications.append(action_vars[agent_id][t] == required_action)

        if implications:
          constraints.append(Implies(is_state_match, And(implications)))

      # Record a summary row for debugging/logging (only once per state)
      summary.append((state_tuple, best_joint_action, fixed_agents))

    logger.info(
      "Strategy constraints built | states=%s | constraints=%s | fixed_agents=%s",
      len(summary),
      len(constraints),
      fixed_agents,
    )

    if ctx.config.debug.dump_strategy_constraints:
      out_path = Path("artifacts/strategy_constraints.txt")
      out_path.parent.mkdir(parents=True, exist_ok=True)
      with open(out_path, "w") as f:
        f.write("state | best_joint_action | fixed_agents\n")
        for row in summary:
          f.write(f"{row[0]} | {row[1]} | {row[2]}\n")
      logger.info("Strategy summary written to %s", out_path)

    return constraints
