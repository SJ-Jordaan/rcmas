"""Z3 variable creation for the RCMAS system."""

from z3 import Int
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
