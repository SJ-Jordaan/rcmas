"""Z3 variable creation for the RCMAS system."""
from itertools import combinations, product

from z3 import Int, Bool, Or, And, Exists, If, Sum

from .utils import sector_to_coords
from .config import NUM_SECTORS, NUM_AGENTS, NUM_TIMESTEPS, INACCESSIBLE_SECTORS

def encode_state():
  S = range(NUM_SECTORS)
  T = range(NUM_TIMESTEPS + 1)

  return [
    [Int(f"sector_{s}_{t}") for t in T]
    for s in S
  ]

def encode_action():
  T = range(NUM_TIMESTEPS)
  A = range(NUM_AGENTS)

  return [
    [Int(f"action_{a}_{t}") for t in T]
    for a in A
  ]

def adj(s1, s2, a):
  return Bool(f"adj_{s1}_{s2}_{a}")

def cr(s1, s2, a):
  return Bool(f"cohesive_relation_{s1}_{s2}_{a}")

def encode_size():
  S = range(NUM_SECTORS)
  A = range(NUM_AGENTS)

  size_vars = [
    [Int(f"size_{i}_{a}") for a in A]
    for i in S
  ]

  return size_vars

def encode_payoff():
  A = range(NUM_AGENTS)

  payoff = [Int(f"payoff_{a}") for a in A]

  return payoff

def encode_variables():
  return encode_state(), encode_action()

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
  T = range(NUM_SECTORS)
  Agt = range(NUM_AGENTS)

  constraints = []

  for i, j in combinations(T, 2):
    for a in Agt:
      xi, yi = sector_to_coords(i)
      xj, yj = sector_to_coords(j)

      are_physically_adjacent = Or(
        And(xi == xj, abs(yi - yj) == 1),
        And(yi == yj, abs(xi - xj) == 1)
      )

      both_owned_by_a = And(
        state[i][NUM_TIMESTEPS] == (a + 1),
        state[j][NUM_TIMESTEPS] == (a + 1)
      )

      constraints.append(
        adj(i, j, a) == And(are_physically_adjacent, both_owned_by_a)
      )

  return constraints

def encode_transitivity():
  T = range(NUM_SECTORS)
  Agt = range(NUM_AGENTS)

  constraints = []

  for (i, j), a in product(combinations(T, 2), Agt):
    k = Int(f"k")

    constraints.append(
        cr(i, j, a) == Or(
          adj(i, j, a),
          Exists([k], And(
            cr(i, k, a),
            cr(k, j, a)
          ))
        )
      )

  return constraints

def encode_objective(state):
  T = range(NUM_SECTORS)
  Agt = range(NUM_AGENTS)

  size_vars = encode_size()
  payoff_vars = encode_payoff()

  objective_constraints = []

  for i in T:
    for a in Agt:
      agent_id = a + 1

      # 1 + Σ cr_{i,j,a}
      cr_sum_terms = [
        If(cr(i, j, a), 1, 0)
        for j in T if i != j
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

  for a in Agt:
    # --- Implement payoff_{a} = max(size_{i,a}) ---
    # This is modeled by two constraints:
    # 1. payoff must be >= all sizes
    # 2. payoff must be == at least one size

    agent_size_vars = [size_vars[i][a] for i in T]

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

