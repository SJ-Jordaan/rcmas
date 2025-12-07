# src/logic/strategy.py
from z3 import And, Implies
from logic.base import BaseConstraint
from core.state import PipelineContext, PipelineMode

class FixedStrategyConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    if not ctx.q_table:
      return []

    constraints = []
    state_vars = ctx.z3_vars['state']
    action_vars = ctx.z3_vars['action']

    fixed_agents = []
    if ctx.mode == PipelineMode.EVAL_BASELINE:
      fixed_agents = list(range(ctx.config.agents.count))
    elif ctx.mode == PipelineMode.AGENT_OPTIMIZATION:
      fixed_agents = [a for a in range(ctx.config.agents.count) if a != ctx.target_agent_idx]

    for state_tuple, action_map in ctx.q_table.items():

      # Find the BEST Joint Action for this state
      # action_map is Dict[JointActionTuple, Value]
      # We want the Key with the Max Value
      if not action_map: continue

      best_joint_action = max(action_map, key=action_map.get)

      # If the best action is still "bad" (-1e9), we effectively ignore this state
      # because the solver naturally avoids invalid moves.
      if action_map[best_joint_action] < -500000:
        continue

      # Add constraints for every timestep
      for t in range(ctx.num_timesteps):

        # 1. Condition: State Matches
        is_state_match = And([
          state_vars[s][t] == val
          for s, val in enumerate(state_tuple)
        ])

        # 2. Consequence: Fixed Agents take their part of the Joint Action
        implications = []
        for agent_id in fixed_agents:
          required_action = best_joint_action[agent_id]
          implications.append(action_vars[agent_id][t] == required_action)

        # 3. Pruning: If Optimizing, forbid collision moves (optional optimization)
        # If we are optimizing Agent X, we know the other agents act according to best_joint_action.
        # Agent X shouldn't pick any sector they picked.
        if ctx.mode == PipelineMode.AGENT_OPTIMIZATION:
          target_agent = ctx.target_agent_idx
          # Collect sectors taken by fixed agents
          occupied_by_fixed = {best_joint_action[a] for a in fixed_agents}

          for forbidden_sector in occupied_by_fixed:
            implications.append(action_vars[target_agent][t] != forbidden_sector)

        if implications:
          constraints.append(Implies(is_state_match, And(implications)))

    return constraints
