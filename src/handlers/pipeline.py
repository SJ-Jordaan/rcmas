import logging
from copy import deepcopy
from typing import List, Dict, Any

from src.core.state import PipelineContext, PipelineMode
from src.handlers.encoder import EncodingHandler
from src.handlers.solver import SolverHandler
from src.handlers.output import OutputHandler
from src.handlers.learner import ensure_q_table, _extract_path, _get_total_payoff, _get_per_agent_payoffs, update_q_from_results


def _reset_ctx(ctx: PipelineContext, mode: PipelineMode, target_agent=None) -> PipelineContext:
    new_ctx = deepcopy(ctx)
    new_ctx.mode = mode
    new_ctx.target_agent_idx = target_agent
    new_ctx.z3_vars = {}
    new_ctx.z3_optimizer = None
    new_ctx.z3_model = None
    new_ctx.found_models = []
    new_ctx.is_satisfiable = False
    new_ctx.last_path = []
    new_ctx.last_payoff = 0.0
    new_ctx.last_q_changes = []
    return new_ctx


def run_episode(ctx: PipelineContext) -> PipelineContext:
    logger = logging.getLogger("rcmas.pipeline")

    ensure_q_table(ctx, logger)

    # Variants: baseline all fixed, plus one-free-agent variants
    variants = [("all_fixed", PipelineMode.EVAL_BASELINE, None)] + [
        (f"agent_{i}_free", PipelineMode.AGENT_OPTIMIZATION, i)
        for i in range(ctx.config.agents.count)
    ]

    results: List[Dict[str, Any]] = []

    for name, mode, target in variants:
        vctx = _reset_ctx(ctx, mode, target)
        logger.info("Running variant=%s mode=%s target=%s", name, mode.value, target)

        # Encode and solve
        vctx = EncodingHandler().handle(vctx)
        vctx = SolverHandler().handle(vctx)

        if vctx.is_satisfiable and vctx.z3_model:
            payoff = _get_total_payoff(vctx)
            per_agent_payoffs = _get_per_agent_payoffs(vctx)
            path = _extract_path(vctx)
        else:
            payoff = float("-inf")
            per_agent_payoffs = None
            path = []

        results.append(
            {
                "name": name,
                "mode": mode,
                "target_agent": target,
                "satisfiable": vctx.is_satisfiable,
                "payoff": payoff,
                "payoffs": per_agent_payoffs,
                "path": path,
                "context": vctx,  # for optional output use
            }
        )

        # Persist each variant's results for inspection
        if vctx.is_satisfiable:
          OutputHandler().handle(vctx)

    # Aggregation + Q-table update
    update_q_from_results(ctx, results)

    # NE check: if no agent can improve over baseline
    eps = ctx.config.debug.epsilon
    baseline = next((r for r in results if r.get("name") == "all_fixed"), None)
    baseline_sat = baseline and baseline.get("satisfiable")
    baseline_payoff = baseline.get("payoff", float("-inf")) if baseline else float("-inf")
    baseline_payoffs = baseline.get("payoffs") if baseline else None

    improvements = []
    if baseline_sat and baseline_payoffs is not None:
        for r in results:
            if r.get("name") == "all_fixed":
                continue
            if not r.get("satisfiable"):
                continue
            target = r.get("target_agent")
            r_payoffs = r.get("payoffs")
            if r_payoffs is None or target is None:
                continue
            if r_payoffs[target] > baseline_payoffs[target] + eps:
                improvements.append(r)

    if baseline_sat and not improvements:
        logger.info("Nash equilibrium detected: no unilateral improvements over baseline.")
        ctx.ne_found = True
        ctx.terminated = True
    elif not baseline_sat:
        logger.warning("Baseline unsatisfiable; marking terminated (no feasible joint strategy).")
        ctx.ne_found = False
        ctx.terminated = True

    return ctx
