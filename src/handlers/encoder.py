from z3 import Optimize, Sum
from src.handlers.base import Handler
from src.core.state import PipelineContext
from src.logic import variables
from src.logic.constraints import (
  ActionAvailabilityConstraint,
  EvolutionConstraint,
  AdjacencyConstraint,
  TransitivityConstraint,
  FullBoardConstraint,
  PayoffConstraint, InitialStateConstraint
)


class SATEncodingHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    print("Initializing Variables...")

    # 1. Variable Phase (Create the vocabulary)
    state, action = variables.create_base_variables(
      ctx.num_sectors, ctx.num_timesteps, ctx.config.agents.count
    )
    adj, cr = variables.create_topology_variables(
      ctx.num_sectors, ctx.config.agents.count
    )
    size, payoff = variables.create_objective_variables(
      ctx.num_sectors, ctx.config.agents.count
    )

    # Store in context for Constraints to access
    ctx.z3_vars = {
      'state': state,
      'action': action,
      'adj': adj,
      'cr': cr,
      'size': size,
      'payoff': payoff
    }

    # 2. Constraint Phase (Define the rules)
    # We can easily add/remove constraints here based on config flags if we wanted!
    active_constraints = [
      InitialStateConstraint(),
      ActionAvailabilityConstraint(),
      EvolutionConstraint(),
      AdjacencyConstraint(),
      TransitivityConstraint(),
      FullBoardConstraint(),
      PayoffConstraint()
    ]

    optimizer = Optimize()

    print("Applying Constraints...")
    for constraint in active_constraints:
      # Each constraint looks at ctx.z3_vars, builds logic, and returns list
      rules = constraint.build(ctx)
      optimizer.add(rules)

    # 3. Objective Phase
    # Maximize the sum of all agent payoffs
    total_payoff = Sum(payoff)
    optimizer.maximize(total_payoff)

    ctx.z3_optimizer = optimizer
    return ctx
