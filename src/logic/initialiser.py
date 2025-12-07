from abc import ABC, abstractmethod
from itertools import product
from typing import Dict, Tuple
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
    num_agents = ctx.config.agents.count
    num_sectors = ctx.num_sectors
    max_timesteps = ctx.num_timesteps
    inaccessible = set(ctx.config.grid.inaccessible_sectors)

    q_table = {}

    # 1. Prepare State Space Iterables
    # Each sector can be [-1] (inaccessible), or [0..N] (empty or agent ID)
    sector_pools = []
    for s in range(num_sectors):
      if s in inaccessible:
        sector_pools.append([-1])
      else:
        sector_pools.append(list(range(num_agents + 1)))

    # 2. Prepare Action Space (Agents move to accessible sectors)
    valid_targets = [s for s in range(num_sectors) if s not in inaccessible]
    all_joint_actions = list(product(valid_targets, repeat=num_agents))

    print(f"[Init] Generating state space for {num_sectors} sectors, {num_agents} agents...")

    # 3. Iterate Cartesian Product of States
    count = 0

    for state in product(*sector_pools):
      # --- VALIDITY CHECK 1: Agent Uniqueness ---
      # Count occurrences of each agent ID (1..N)
      counts = [0] * (num_agents + 1)
      occupied_count = 0

      for val in state:
        if val > 0:
          counts[val] += 1
          occupied_count += 1

      # Constraint: Each agent appears AT MOST once.
      if any(c > 1 for c in counts[1:]):
        continue

      # Constraint: Agents appear uniformly (Sync movement constraint)
      # If Agent 1 is on board, Agent 2 must be on board (assuming sync start)
      # This forces sum(counts[1:]) == occupied_count where all c are 1
      # Which is covered by the next check.

      # --- VALIDITY CHECK 2: Time/Occupancy Invariant ---
      # "Exactly one per sector must be acquired per agent per timestep"
      # Therefore: occupied_count must be a multiple of num_agents
      if occupied_count % num_agents != 0:
        # Impossible state (e.g. 1 agent on board, 1 agent off board)
        # We skip adding this to the table entirely to save memory/time
        continue

      # Check Max Time bounds
      current_t = occupied_count // num_agents
      if current_t > max_timesteps:
        continue

      # Initialize State Entry
      q_table[state] = {}

      # Populate Actions
      for action in all_joint_actions:
        # --- VALIDITY CHECK 3: Action Collisions ---
        # "Ignore state-action pairs where two or more agents take same action"
        if len(set(action)) < num_agents:
          # Penalize heavily (Mission Failure)
          q_table[state][action] = -1_000_000.0
        else:
          # Valid move, neutral start
          q_table[state][action] = 0.0

      count += 1

    print(f"[Init] Q-Table generated with {count} reachable states.")
    return q_table
