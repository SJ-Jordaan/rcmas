"""Z3 variable creation for the RCMAS system."""
from itertools import combinations, permutations

from z3 import Int, Bool, Or, And, Implies

from .utils import sector_to_coords
from .config import NUM_SECTORS, NUM_AGENTS, NUM_TIMESTEPS, INACCESSIBLE_SECTORS


def encode_variables():
  state = [
    [Int(f"sector_{s}_{t}") for t in range(NUM_TIMESTEPS + 1)]
    for s in range(NUM_SECTORS)
  ]

  action = [
    [Int(f"action_{a}_{t}") for t in range(NUM_TIMESTEPS)]
    for a in range(NUM_AGENTS)
  ]

  return state, action


def encode_initial_state(state):
  inaccessible_constraints = [
    state[s][t] == -1
    for s in INACCESSIBLE_SECTORS
    for t in range(NUM_TIMESTEPS + 1)
  ]

  initial_empty_constraints = [
    state[s][0] == 0
    for s in range(NUM_SECTORS) if s not in INACCESSIBLE_SECTORS
  ]

  return inaccessible_constraints + initial_empty_constraints

def encode_adjacency(state):
  S = range(NUM_SECTORS)  # 0 to S-1
  A = range(NUM_AGENTS)  # 0 to A-1

  adjacency = []

  for i, j in permutations(S, 2):
    for a in A:
      x, y = sector_to_coords(i)
      x_prime, y_prime = sector_to_coords(j)
      adjacency.append(
        Bool(f"adjacent_{i}_{j}_{a}") == Or(
          And(
            x == x_prime - 1,
            y == y_prime,
            state[i][NUM_TIMESTEPS] == (a + 1),
            state[j][NUM_TIMESTEPS] == (a + 1)
          ),
          And(
            x == x_prime + 1,
            y == y_prime,
            state[i][NUM_TIMESTEPS] == (a + 1),
            state[j][NUM_TIMESTEPS] == (a + 1)
          ),
          And(
            y == y_prime - 1,
            x == x_prime,
            state[i][NUM_TIMESTEPS] == (a + 1),
            state[j][NUM_TIMESTEPS] == (a + 1)
          ),
          And(
            y == y_prime + 1,
            x == x_prime,
            state[i][NUM_TIMESTEPS] == (a + 1),
            state[j][NUM_TIMESTEPS] == (a + 1)
          ),
        )
      )

      adjacency.append(
        Bool(f"cohesive_relation_{i}_{j}_{a}") == Bool(f"adjacent_{i}_{j}_{a}")
      )

  return adjacency

def encode_transitivity():
  S = range(NUM_SECTORS)  # 0 to S-1
  A = range(NUM_AGENTS)  # 0 to A-1

  transitivity = []

  for i, j, k in permutations(S, 3):
    for a in A:
      transitivity.append(
        Implies(
          And(
            Bool(f"cohesive_relation_{i}_{j}_{a}"),
            Bool(f"cohesive_relation_{j}_{k}_{a}")
          ),
          Bool(f"cohesive_relation_{i}_{k}_{a}")
        )
      )

  return transitivity
