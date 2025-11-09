"""Z3 variable creation for the RCMAS system."""
from itertools import permutations

from z3 import Int, Bool, Or, And, Implies, If, Sum

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
        Implies(
          Bool(f"adjacent_{i}_{j}_{a}"),
          Bool(f"cohesive_relation_{i}_{j}_{a}")
        )
      )

      adjacency.append(
        Implies(
          Bool(f"cohesive_relation_{i}_{j}_{a}"),
          And(
            state[i][NUM_TIMESTEPS] == (a + 1),
            state[j][NUM_TIMESTEPS] == (a + 1)
          )
        )
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


def encode_objective(state):
  """
  Implements the size and payoff logic from the LaTeX spec.
  """
  S = range(NUM_SECTORS)
  A = range(NUM_AGENTS)

  # --- Define size_{i,a} variables ---
  size_vars = [
    [Int(f"size_{i}_{a}") for a in A]
    for i in S
  ]

  # --- Define payoff_{a} variables ---
  payoff_vars = [Int(f"payoff_{a}") for a in A]

  objective_constraints = []

  for i in S:
    for a in A:
      agent_id = a + 1

      # Create the sum: 1 + Σ cr_{i,j,a}
      # Use If(cr_var, 1, 0) to treat Bool as Int
      cr_sum_terms = [
        If(Bool(f"cohesive_relation_{i}_{j}_{a}"), 1, 0)
        for j in S if i != j
      ]

      # Implement the rule:
      # size_{i,a} =
      #   IF (i is not 'a') THEN 0
      #   ELSE (1 + sum(cr_{i,j,a}))
      objective_constraints.append(
        If(
          state[i][NUM_TIMESTEPS] != agent_id,
          size_vars[i][a] == 0,
          size_vars[i][a] == (1 + Sum(cr_sum_terms))
        )
      )

  for a in A:
    # --- Implement payoff_{a} = max(size_{i,a}) ---
    # This is modeled by two constraints:
    # 1. payoff must be >= all sizes
    # 2. payoff must be == at least one size

    agent_size_vars = [size_vars[i][a] for i in S]

    objective_constraints.extend(
      [payoff_vars[a] >= size for size in agent_size_vars]
    )

    objective_constraints.append(
      Or([payoff_vars[a] == size for size in agent_size_vars])
    )

  # --- Define the final objective function ---
  # max Σ payoff_{a}
  total_payoff = Sum(payoff_vars)

  return objective_constraints, total_payoff
