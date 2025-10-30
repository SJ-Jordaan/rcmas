"""Solver setup and execution for the RCMAS system."""

from z3 import Solver, Or, sat
from .variables import create_sector_variables, create_action_variables, create_initial_state
from .constraints import (
    generate_all_constraints
)


def setup_solver():
    """
    Create and configure a Z3 solver with all constraints.
    
    Returns:
        Solver: Configured Z3 solver instance
    """
    solver = Solver()
    
    initial_state = create_initial_state()
    constraints = generate_all_constraints()
    
    solver.add(initial_state)
    solver.add(constraints)
    
    return solver


def find_solution(solver):
    """
    Check if a solution exists and return it.
    
    Args:
        solver: Configured Z3 solver instance
        
    Returns:
        model or None: Z3 model if satisfiable, None otherwise
    """
    if solver.check() == sat:
        return solver.model()
    return None


def block_solution(solver, model):
    """
    Add constraints to block the current solution, allowing search for new solutions.
    
    Args:
        solver: Z3 solver instance
        model: Current model to block
    """
    block = []
    
    sectors = create_sector_variables()
    for sector_var in sectors:
        block.append(sector_var != model.eval(sector_var, model_completion=True))
    
    # Block action variables
    actions = create_action_variables()
    for action_var in actions:
        block.append(action_var != model.eval(action_var, model_completion=True))
    
    solver.add(Or(block))
