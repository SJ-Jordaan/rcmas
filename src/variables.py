"""Z3 variable creation for the RCMAS system."""
from itertools import combinations, product

from z3 import Int, Bool, Or, And, Not, If, Sum

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
    constraints.append(
        cr(i, j, a) == Or(
          adj(i, j, a),
          Or(
            And(
              cr(i,k,a),
              cr(k,j,a)
            )
            for k in T if i < k < j
          )
        )
      )

    constraints.append(
        cr(i, j, a) == And(
          adj(i, j, a),
          And(
            Or(
              Not(cr(i,k,a)),
              Not(cr(k,j,a))
            )
            for k in T if i < k < j
          )
        )
      )

  return constraints
