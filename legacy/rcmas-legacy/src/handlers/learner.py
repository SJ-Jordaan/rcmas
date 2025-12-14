import logging
from pathlib import Path
from logic.initialiser import MinimalQTableInitializer, build_action_map
from handlers.base import Handler
from core.state import PipelineContext, PipelineMode
from utils.debug import q_deltas, format_changes_table, serialize_q_table

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


def _extract_transitions(ctx):
  """Helper to get [(state_t, action_t, state_t1, terminal)] from Z3 model."""
  model = ctx.z3_model
  state_vars = ctx.z3_vars['state']
  action_vars = ctx.z3_vars['action']
  transitions = []

  for t in range(ctx.num_timesteps):
    s_t = tuple(model.eval(state_vars[s][t], model_completion=True).as_long()
                for s in range(ctx.num_sectors))
    a_t = tuple(model.eval(action_vars[a][t], model_completion=True).as_long()
                for a in range(ctx.config.agents.count))
    s_t1 = tuple(model.eval(state_vars[s][t + 1], model_completion=True).as_long()
                 for s in range(ctx.num_sectors))
    terminal = t == ctx.num_timesteps - 1
    transitions.append((s_t, a_t, s_t1, terminal))

  return transitions


def _count_occupied(state):
  """Count occupied sectors (positive agent ids)."""
  return sum(1 for v in state if v > 0)


def _max_cluster_sizes(state, width):
  """Compute largest connected component size per agent (4-neighbor)."""
  agent_cells = {}
  for idx, v in enumerate(state):
    if v <= 0:
      continue
    agent_cells.setdefault(v, set()).add(idx)

  def neighbors(i):
    r, c = divmod(i, width)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      nr, nc = r + dr, c + dc
      if nr < 0 or nc < 0:
        continue
      j = nr * width + nc
      yield j

  max_sizes = {}
  for agent, cells in agent_cells.items():
    seen = set()
    best = 0
    for cell in cells:
      if cell in seen:
        continue
      stack = [cell]
      comp = 0
      while stack:
        cur = stack.pop()
        if cur in seen or cur not in cells:
          continue
        seen.add(cur)
        comp += 1
        for nb in neighbors(cur):
          stack.append(nb)
      best = max(best, comp)
    max_sizes[agent] = best
  return max_sizes


def _compute_reward(ctx, s_t, a_t, s_t1, terminal, episode_payoff):
  """Shaped reward: cohesion gain + gentle payoff shaping + stall penalty.

  - Per-step base: increase in largest cluster sizes across agents.
  - Shaping: add a small fraction of the episode payoff each step.
  - Stall penalty: small negative if no cohesion gain.
  - Terminal bonus: full episode payoff at the last step.
  """
  width = ctx.config.grid.width
  sizes_t = _max_cluster_sizes(s_t, width)
  sizes_t1 = _max_cluster_sizes(s_t1, width)

  def total_sizes(sizes):
    return sum(sizes.values()) if sizes else 0

  cohesion_gain = float(total_sizes(sizes_t1) - total_sizes(sizes_t))
  reward = cohesion_gain

  # Mild shaping toward better final solutions
  w_connect = getattr(ctx.config.debug, "reward_connect_weight", 0.0)
  if w_connect:
    steps = max(1, ctx.num_timesteps)
    reward += w_connect * float(episode_payoff) / steps

  # Penalize stalling in cohesion
  if cohesion_gain == 0.0:
    stall_pen = getattr(ctx.config.debug, "reward_stall_penalty", 0.0)
    reward -= float(stall_pen)

  if terminal:
    reward += float(episode_payoff)
  return reward


def _get_total_payoff(ctx):
  model = ctx.z3_model
  payoff_vars = ctx.z3_vars['payoff']
  return sum(model.eval(p, model_completion=True).as_long() for p in payoff_vars)

def _get_per_agent_payoffs(ctx):
  model = ctx.z3_model
  payoff_vars = ctx.z3_vars['payoff']
  return [model.eval(p, model_completion=True).as_long() for p in payoff_vars]

def ensure_q_table(ctx: PipelineContext, logger: logging.Logger):
  if ctx.q_table:
    return
  logger.info("Q-Table empty, initializing")
  initializer = MinimalQTableInitializer()
  ctx.q_table = initializer.generate(ctx)
  logger.info("Q-Table initialized with %s states", len(ctx.q_table))
  if ctx.config.debug.dump_q_json:
    artifact_path = serialize_q_table(ctx.q_table, Path("artifacts/q_table_initial.json"))
    logger.info("Initial Q-table dumped to %s", artifact_path)


class LearningHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    logger = logging.getLogger("rcmas.learning")
    ensure_q_table(ctx, logger)
    ctx.mode = PipelineMode.EVAL_BASELINE
    return ctx

def update_q_from_results(ctx: PipelineContext, results: list):
  """Aggregate solver results and update Q-table."""
  logger = logging.getLogger("rcmas.learning")
  if not results:
    logger.warning("No results to update Q-table.")
    return

  prev_q = {s: actions.copy() for s, actions in ctx.q_table.items()}

  alpha = ctx.config.debug.learning_rate
  gamma = ctx.config.debug.discount
  hard_update = getattr(ctx.config.debug, "hard_solver_update", False)
  eps = ctx.config.debug.epsilon

  # Baseline payoff
  baseline_payoff = float("-inf")
  baseline_sat = False
  for r in results:
    if r.get("name") == "all_fixed" and r.get("satisfiable"):
      baseline_payoff = r.get("payoff", float("-inf"))
      baseline_sat = True
      break

  # Apply updates from all satisfiable runs
  for r in results:
    if not r.get("satisfiable"):
      continue
    payoff = r.get("payoff", 0.0)
    vctx = r.get("context")
    if not vctx or not vctx.z3_model:
      continue

    transitions = _extract_transitions(vctx)
    better_policy = False
    if baseline_sat and payoff >= baseline_payoff + eps:
      better_policy = True

    for (s_t, a_t, s_t1, terminal) in transitions:
      # Ensure state/action maps exist
      if s_t not in ctx.q_table:
        action_map = build_action_map(ctx, s_t)
        if not action_map:
          continue
        ctx.q_table[s_t] = action_map

      if a_t not in ctx.q_table[s_t]:
        # Ignore actions not valid for this state
        continue

      if s_t1 not in ctx.q_table:
        action_map_next = build_action_map(ctx, s_t1)
        if action_map_next:
          ctx.q_table[s_t1] = action_map_next

      max_next = 0.0
      if not terminal and s_t1 in ctx.q_table and ctx.q_table[s_t1]:
        max_next = max(ctx.q_table[s_t1].values())

      reward = _compute_reward(ctx, s_t, a_t, s_t1, terminal, payoff)
      old_q = ctx.q_table[s_t].get(a_t, 0.0)
      target = reward + (gamma * max_next if not terminal else 0.0)
      if hard_update and better_policy:
        # Trust solver-derived policies that beat the current baseline
        ctx.q_table[s_t][a_t] = max(old_q, target)
      else:
        new_q = (1 - alpha) * old_q + alpha * target
        ctx.q_table[s_t][a_t] = new_q

    if r.get("name") != "all_fixed":
      agent = r.get("target_agent")
      if payoff > baseline_payoff:
        logger.info("Agent %s improved over baseline: %s -> %s", agent, baseline_payoff, payoff)

  # Track and dump changes
  eps = ctx.config.debug.epsilon
  limit = ctx.config.debug.max_q_deltas
  changes = q_deltas(prev_q, ctx.q_table, epsilon=eps, limit=limit)
  ctx.last_q_changes = changes
  if changes:
    table = format_changes_table(changes, max_rows=limit)
    logger.debug("Q-table updates (capped to %s):\n%s", limit, table)
  else:
    logger.debug("No significant Q-table changes (eps=%s)", eps)

  if ctx.config.debug.dump_q_json:
    artifact_path = serialize_q_table(ctx.q_table, Path("artifacts/q_table.json"))
    logger.info("Q-table dumped to %s", artifact_path)
