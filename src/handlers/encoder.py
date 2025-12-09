import logging
from pathlib import Path
from z3 import Optimize, Sum
from handlers.base import Handler
from core.state import PipelineContext, PipelineMode
from logic import variables
from logic.constraints import (
  ActionAvailabilityConstraint,
  EvolutionConstraint,
  AdjacencyConstraint,
  FullBoardConstraint,
  PayoffConstraint,
  InitialStateConstraint,
  TransitivityConstraint
)
from logic.strategy import FixedStrategyConstraint

class EncodingHandler(Handler):
  def handle(self, ctx: PipelineContext) -> PipelineContext:
    logger = logging.getLogger("rcmas.encoder")
    logger.info("Initializing Z3 variables")

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

    include_strategy = ctx.config.debug.include_strategy_constraints

    # 2. Constraint Phase (Define the rules)
    # Strategy constraints can be disabled to diagnose encoding vs. policy.
    active_constraints = [
      InitialStateConstraint(),
      ActionAvailabilityConstraint(),
      EvolutionConstraint(),
      AdjacencyConstraint(),
      TransitivityConstraint(),
      FullBoardConstraint(),
      PayoffConstraint()
    ]

    if include_strategy and ctx.q_table:
      active_constraints.append(FixedStrategyConstraint())
    else:
      logger.info(
        "Skipping FixedStrategyConstraint (include_strategy=%s, q_table=%s)",
        include_strategy,
        bool(ctx.q_table),
      )

    optimizer = Optimize()

    logger.info("Applying constraints (%s blocks)", len(active_constraints))
    total_rules = 0
    for constraint in active_constraints:
      rules = constraint.build(ctx)
      optimizer.add(rules)
      total_rules += len(rules)

    logger.info("Constraint application complete | total_rules=%s", total_rules)

    # 3. Objective Phase
    # Maximize either the system-wide sum or a specific agent's payoff in agent_opt mode
    total_payoff = Sum(payoff)
    if (
      ctx.mode == PipelineMode.AGENT_OPTIMIZATION
      and ctx.target_agent_idx is not None
    ):
      optimizer.maximize(payoff[ctx.target_agent_idx])
    else:
      optimizer.maximize(total_payoff)

    # Optional: dump constraints for inspection
    if ctx.config.debug.dump_constraints:
      out_dir = Path("artifacts")
      out_dir.mkdir(parents=True, exist_ok=True)
      name = f"constraints_{ctx.mode.value}_target_{ctx.target_agent_idx}.smt2"
      out_path = out_dir / name

      # Collect assertions and objective
      assertions = optimizer.assertions()  # type: ignore
      with open(out_path, "w") as f:
        f.write(f"; mode={ctx.mode.value} target={ctx.target_agent_idx}\n")
        f.write(f"; total_rules={total_rules}\n")
        for a in assertions:  # type: ignore
          try:
            a_str = a.sexpr()  # type: ignore[attr-defined]
          except Exception:
            a_str = str(a)
          f.write(a_str)
          f.write("\n")
        f.write("; objective\n")
        try:
          f.write(total_payoff.sexpr())  # type: ignore[attr-defined]
        except Exception:
          f.write(str(total_payoff))
        f.write("\n")

      logger.info("Constraints dumped to %s", out_path)

    ctx.z3_optimizer = optimizer
    return ctx
