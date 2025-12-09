from abc import ABC, abstractmethod
from itertools import product
from typing import Dict, Tuple, List
from core.state import PipelineContext


class QTableInitializer(ABC):
  """Strategy interface for initializing the Q-Table."""

  @abstractmethod
  def generate(self, ctx: PipelineContext) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]]:
    pass


class MinimalQTableInitializer(QTableInitializer):
  """
  Generates a single deterministic game trajectory for the Q-Table.

  Starting from the initial state it walks forward, picking one valid joint
  action per timestep (lexicographically first) and recording only that
  action for the encountered state. No branching or breadth/depth expansion.
  """

  def generate(self, ctx: PipelineContext) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]]:
    """Generate a single plausible path of length = timesteps (or until blocked)."""

    num_agents = ctx.config.agents.count
    num_sectors = ctx.num_sectors
    inaccessible = set(ctx.config.grid.inaccessible_sectors)
    horizon = ctx.num_timesteps or (num_sectors // num_agents)

    # Cap horizon so we never plan more joint moves than accessible cells allow.
    accessible_cells = num_sectors - len(inaccessible)
    max_fill_steps = accessible_cells // num_agents
    horizon = min(horizon, max_fill_steps)

    # Initial state: inaccessible = -1, everything else empty (0)
    init_state: List[int] = []
    for s in range(num_sectors):
      if s in inaccessible:
        init_state.append(-1)
      else:
        init_state.append(0)
    init_state_tuple = tuple(init_state)

    q_table: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {}

    current_state = init_state_tuple

    for _ in range(horizon):
      action_map = build_action_map(ctx, current_state)
      if not action_map:
        break

      # Pick a single deterministic joint action (lexicographically first)
      chosen_action = min(action_map.keys())
      q_table[current_state] = {chosen_action: 0.0}

      next_state = step_state(current_state, chosen_action, num_agents)
      current_state = next_state

    print(f"[Init] Q-Table generated with {len(q_table)} states on a single trajectory (up to {horizon} steps).")
    return q_table


def build_action_map(ctx: PipelineContext, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], float]:
  """Build valid joint actions for a given state lazily.

  Rules:
  - No collisions (distinct targets per agent)
  - No moves into already occupied sectors
  """
  num_agents = ctx.config.agents.count
  inaccessible = set(ctx.config.grid.inaccessible_sectors)
  num_sectors = ctx.num_sectors

  valid_targets = [s for s in range(num_sectors) if s not in inaccessible]
  all_joint_actions = product(valid_targets, repeat=num_agents)

  occupied_sectors = {idx for idx, val in enumerate(state) if val > 0}

  action_map: Dict[Tuple[int, ...], float] = {}
  for action in all_joint_actions:
    # Collision: two or more agents pick same target
    if len(set(action)) < num_agents:
      continue

    # Any agent tries to move into an already occupied sector?
    if any(target in occupied_sectors for target in action):
      continue

    action_map[tuple(action)] = 0.0

  return action_map


def step_state(state: Tuple[int, ...], action: Tuple[int, ...], num_agents: int) -> Tuple[int, ...]:
  """Apply the occupancy-accretion transition used in the constraints.

  Each agent permanently occupies its chosen target; we assume the caller
  already filtered collisions and moves into occupied sectors.
  """
  new_state = list(state)
  for agent_idx, target in enumerate(action):
    agent_id = agent_idx + 1
    new_state[target] = agent_id
  return tuple(new_state)
