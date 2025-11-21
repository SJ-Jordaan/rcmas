"""Solver setup and execution for the RCMAS system."""

from z3 import Solver, Optimize, Or, sat, set_option

from config import NUM_SECTORS, NUM_TIMESTEPS, NUM_AGENTS
from .variables import (
  encode_variables,
  encode_initial_state,
  encode_adjacency,
  encode_transitivity,
)
from .constraints import (
  encode_action_availability,
  encode_evolution,
  encode_full_board
)

def setup_solver(use_soft_constraints=True):
  set_option(verbose=10)

  state, action = encode_variables()
  adjacency = encode_adjacency(state)
  transitivity = encode_transitivity()
  initial_state = encode_initial_state(state)
  action_availability = encode_action_availability(state, action)
  evolution = encode_evolution(state, action)
  full_board = encode_full_board(state)

  soft_constraints = []
  if use_soft_constraints:
    optimizer = Optimize()
    optimizer.add(initial_state)
    optimizer.add(adjacency)
    optimizer.add(transitivity)
    optimizer.add(action_availability)
    optimizer.add(evolution)
    optimizer.add(full_board)

    for soft in soft_constraints:
      optimizer.add_soft(soft)

    return optimizer
  else:
    solver = Solver()
    solver.add(initial_state)
    solver.add(adjacency)
    solver.add(action_availability)
    solver.add(transitivity)
    solver.add(evolution)
    solver.add(full_board)
    return solver


def find_solution(solver):
  if solver.check() == sat:
    return solver.model()
  return None


def block_solution(solver, model, state, action):
  # Get all state variables (state[s][0]...state[s][T])
  all_state_vars = [
    state[s][t]
    for s in range(NUM_SECTORS)
    for t in range(NUM_TIMESTEPS + 1)  # Use T+1 for state
  ]

  # Get all action variables (action[a][0]...action[a][T-1])
  all_action_vars = [
    action[a][t]
    for a in range(NUM_AGENTS)
    for t in range(NUM_TIMESTEPS)  # Use T for action
  ]

  all_vars = all_state_vars + all_action_vars

  block = [
    var != model.eval(var, model_completion=True)
    for var in all_vars
  ]

  # This means "at least one variable must change"
  solver.add(Or(block))
