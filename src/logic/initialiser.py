from abc import ABC, abstractmethod
from itertools import product
from typing import Dict, Tuple, List, Set
from core.state import PipelineContext


class QTableInitializer(ABC):
  """Strategy interface for initializing the Q-Table."""

  @abstractmethod
  def generate(self, ctx: PipelineContext) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]]:
    pass


class ValidStateInitializer(QTableInitializer):
  """
  Generates a Joint Q-Table but pre-penalizes:
  1. Collision Actions (multiple agents picking same sector)
  2. Unreachable States (states that violate the time/occupation invariant)
  """

  def generate(self, ctx: PipelineContext) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]]:
    """Generate reachable states up to the timestep horizon using forward expansion.

    This avoids full Cartesian explosion while giving the strategy encoder
    meaningful initial support.
    """

    num_agents = ctx.config.agents.count
    num_sectors = ctx.num_sectors
    inaccessible = set(ctx.config.grid.inaccessible_sectors)
    horizon = ctx.num_timesteps or (num_sectors // num_agents)

    # Initial state: inaccessible = -1, everything else empty (0)
    init_state: List[int] = []
    for s in range(num_sectors):
      if s in inaccessible:
        init_state.append(-1)
      else:
        init_state.append(0)
    init_state_tuple = tuple(init_state)

    q_table: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {}

    frontier: List[Tuple[int, Tuple[int, ...]]] = [(0, init_state_tuple)]
    visited: Set[Tuple[int, ...]] = set()

    while frontier:
      t, state = frontier.pop()
      if state in visited:
        continue
      visited.add(state)

      action_map = build_action_map(ctx, state)
      if action_map:
        q_table[state] = action_map

      # Do not expand beyond horizon
      if t >= horizon:
        continue

      # Expand successors
      for action in action_map.keys():
        next_state = step_state(state, action, num_agents)
        frontier.append((t + 1, next_state))

      print(f"[Init] Q-Table generated with {len(q_table)} reachable states (up to {horizon} steps).")
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
