"""Solver setup and execution for the RCMAS system."""

from z3 import Solver, Optimize, Or, sat, set_option
from .variables import create_sector_variables, create_action_variables, create_initial_state
from .constraints import (
  generate_all_hard_constraints,
  generate_maximise_cohesive_region_constraints
)


def setup_solver(use_soft_constraints=True):
  """
  Create and configure a Z3 solver with all constraints.

  Args:
      use_soft_constraints: If True, use Optimize with soft constraints for cohesive regions

  Returns:
      Solver or Optimize: Configured Z3 solver/optimizer instance
  """
  set_option(verbose=10)

  initial_state = create_initial_state()
  hard_constraints = generate_all_hard_constraints()

  soft_constraints = []
  if use_soft_constraints:
    soft_constraints = generate_maximise_cohesive_region_constraints()

  if soft_constraints:
    optimizer = Optimize()
    optimizer.add(initial_state)
    optimizer.add(hard_constraints)

    for soft in soft_constraints:
      optimizer.add_soft(soft)

    return optimizer
  else:
    solver = Solver()
    solver.add(initial_state)
    solver.add(hard_constraints)
    return solver


def find_solution(solver):
  """
  Check if a solution exists and return it.
  Works with both Solver and Optimize instances.

  Args:
      solver: Configured Z3 solver or optimizer instance

  Returns:
      model or None: Z3 model if satisfiable, None otherwise
  """
  if solver.check() == sat:
    return solver.model()
  return None


def block_solution(solver, model):
  """
  Add constraints to block the current solution, allowing search for new solutions.
  Works with both Solver and Optimize instances.

  Args:
      solver: Z3 solver or optimizer instance
      model: Current model to block
  """
  block = []

  sectors = create_sector_variables()
  for sector_var in sectors:
    block.append(sector_var != model.eval(sector_var, model_completion=True))

  actions = create_action_variables()
  for action_var in actions:
    block.append(action_var != model.eval(action_var, model_completion=True))

  solver.add(Or(block))
