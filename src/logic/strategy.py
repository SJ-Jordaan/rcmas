# src/logic/strategy.py
import logging
from numbers import Number
from pathlib import Path
from z3 import And, Implies
from logic.base import BaseConstraint
from core.state import PipelineContext, PipelineMode
from logic.initialiser import build_action_map

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

    # Build a chosen policy; if current argmax policy was seen, walk states from the tail and pick the first unseen alternative signature.
    policy_actions = {state: max(actions.items(), key=lambda kv: kv[1])[0] for state, actions in ctx.q_table.items() if actions}
    strategy_sig = _policy_signature(policy_actions)

    if strategy_sig in ctx.strategy_signatures:
      logger.info("Strategy already used; searching for the first unseen alternative policy")
      states = list(policy_actions.keys())
      for state in reversed(states):
        valid_actions = build_action_map(ctx, state)
        if not valid_actions:
          continue
        current = policy_actions[state]
        alternatives = [a for a in sorted(valid_actions.keys()) if a != current]
        for alt in alternatives:
          candidate_policy = dict(policy_actions)
          candidate_policy[state] = alt
          candidate_sig = _policy_signature(candidate_policy)
          if candidate_sig not in ctx.strategy_signatures:
            policy_actions = candidate_policy
            strategy_sig = candidate_sig
            break
        if strategy_sig not in ctx.strategy_signatures:
          break

    ctx.strategy_signatures.add(strategy_sig)

    if getattr(ctx.config.debug, "dump_policy_signatures", False):
      out_path = Path("artifacts/policy_signatures.txt")
      out_path.parent.mkdir(parents=True, exist_ok=True)
      with open(out_path, "a") as f:
        f.write(f"signature: {strategy_sig}\n")
        for state, action in sorted(policy_actions.items()):
          f.write(f"  {state} -> {action}\n")
        f.write("---\n")

    summary = []

    for state_tuple, action_map in ctx.q_table.items():

      if not action_map:
        continue

      best_joint_action = policy_actions.get(state_tuple)
      if best_joint_action is None:
        continue

      for agent_id in fixed_agents:
        required_action = best_joint_action[agent_id]
        t = _state_to_timestep(state_tuple, ctx.config.agents.count)
        constraints.append(
          action_vars[agent_id][t] == required_action
        )

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


def _policy_signature(policy_actions: dict) -> str:
  """Stable signature of chosen actions per state."""
  items = sorted(policy_actions.items())
  return str(items)


def _state_to_timestep(state: tuple, count: int) -> int:
  if count == 0: return 0
  return sum(1 for cell in state if cell > 0) // count
