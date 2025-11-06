"""Constraint generation for the RCMAS system."""
from itertools import product, combinations

from z3 import Implies, And, Or

from .config import (
  NUM_SECTORS, NUM_AGENTS, NUM_TIMESTEPS, INACCESSIBLE_SECTORS
)

def encode_action_availability(state, action):
  # --- 1. Define Iteration Ranges ---
  T = range(NUM_TIMESTEPS)  # Timesteps for actions: 0 to T-1
  S = range(NUM_SECTORS)  # Sectors: 0 to S-1
  A = range(NUM_AGENTS)  # Agents (0-indexed): 0 to A-1

  # --- 2. Action Bound Constraints ---
  # 0 <= action[a][t]
  lower_bound_constraints = [
    action[a][t] >= 0
    for t, a in product(T, A)  # "For all timesteps t and agents a"
  ]

  # action[a][t] < NUM_SECTORS
  upper_bound_constraints = [
    action[a][t] < NUM_SECTORS
    for t, a in product(T, A)  # "For all timesteps t and agents a"
  ]

  # --- 3. Sector Availability Constraints ---
  # If a sector 's' at time 't' is not empty (state != 0),
  # THEN no agent 'a' can choose it (action[t][a] != s).
  occupied_sector_constraints = [
    Implies(
      state[s][t] != 0,
      action[a][t] != s
    )
    for t, s, a in product(T, S, A)  # "For all t, s, and a"
  ]

  # --- 4. Return Everything ---
  return (
    lower_bound_constraints +
    upper_bound_constraints +
    occupied_sector_constraints
  )

def encode_evolution(state, action):
  # --- 1. Define Iteration Ranges ---
  T = range(NUM_TIMESTEPS)  # 0 to T-1
  S = range(NUM_SECTORS)  # 0 to S-1
  A = range(NUM_AGENTS)  # 0 to A-1

  constraints = []

  # "For all timesteps t and sectors s..."
  for t, s in product(T, S):
    # --- Case 1: Successful Action ---
    # One Implies() for each agent
    for a in A:
      agent_id = a + 1

      # Condition: This agent 'a' chose 's' AND no other agent did
      unique_action = And(
        action[a][t] == s,
        And([
          action[other_a][t] != s
          for other_a in A if other_a != a
        ])
      )

      constraints.append(
        Implies(unique_action, state[s][t+1] == agent_id)
      )

    # --- Case 2: No Action ---
    # Condition: NO agent chose sector 's'
    no_action = And([
      action[a][t] != s
      for a in A
    ])

    constraints.append(
      Implies(no_action, state[s][t+1] == state[s][t])
    )

    # --- Case 3: Conflict ---
    # Condition: AT LEAST two agents chose 's'
    conflict = Or([
      And(
        action[a1][t] == s,
        action[a2][t] == s
      )
      for a1, a2 in combinations(A, 2)
    ])

    constraints.append(
      Implies(conflict, state[s][t+1] == state[s][t])
    )

  return constraints

def encode_full_board(state):
  accessible_sectors = [
    s for s in range(NUM_SECTORS) if s not in INACCESSIBLE_SECTORS
  ]

  return [
    state[s][NUM_TIMESTEPS] != 0
    for s in accessible_sectors
  ]

# Q learning algorithm per agent
#  combative learning
# state     | action  | q-value
# (0,0,0,0) | 0       | 0         |
# (0,0,0,0) | 1       | 0.343451  |
# (0,0,0,0) | 2       | 0.67575   |
# (0,0,0,0) | 3       | -3.44545  |
# ...
# What is the definition of a Q learning state in this instance?

# Formal Strategy
# state     | a1 | a2 |
# (0,0,0,0) | 2  | 1  |
# (1,0,2,0) | 0  | 2  |
#   ...
# (1,0,0,2) | 2  | 1  |

# If every region is unoccupied -> timestep 0

# The number of occupied regions / number of agents = timestep
