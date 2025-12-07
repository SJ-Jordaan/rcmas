from z3 import Optimize, Sum
from src.handlers.base import Handler
from src.core.state import PipelineContext
from src.logic import variables, constraints

class SATEncodingHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    S = ctx.num_sectors
    T = ctx.num_timesteps
    A = ctx.config.agents.count
    W = ctx.config.grid.width
    inaccessible = ctx.config.grid.inaccessible_sectors

    state, action = variables.encode_variables(S, A, T)
    adjacency = variables.encode_adjacency(state, S, A, T, W)
    transitivity = variables.encode_transitivity(S, A, W)
    initial_state = variables.encode_initial_state(state, inaccessible, T, S)
    action_availability = constraints.encode_action_availability(state, action, S, A, T)
    evolution = constraints.encode_evolution(state, action, S, A, T)
    full_board = constraints.encode_full_board(state, S, T, inaccessible)
    size_vars = variables.encode_size(state, S, A, T)
    payoff_vars, payoff_constraints = variables.encode_payoff(S, A)

    optimizer = Optimize()
    optimizer.add(initial_state)
    optimizer.add(adjacency)
    optimizer.add(transitivity)
    optimizer.add(action_availability)
    optimizer.add(evolution)
    optimizer.add(full_board)
    optimizer.add(size_vars)
    optimizer.add(payoff_constraints)

    objective = Sum(payoff_vars)
    optimizer.maximize(objective)
    ctx.z3_vars = {
      'state': state,
      'action': action,
    }
    ctx.z3_optimizer = optimizer
    return ctx
