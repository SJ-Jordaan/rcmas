# src/main.py
import logging
import sys
from pathlib import Path
from src.core.config import AppConfig
from src.core.state import PipelineContext
from src.handlers.pipeline import run_episode


def _setup_logging(config: AppConfig) -> logging.Logger:
    log_level = getattr(logging, config.debug.level.upper(), logging.INFO)
    logger = logging.getLogger("rcmas")
    logger.setLevel(log_level)

    # Avoid duplicate handlers if main is re-run in the same process
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logging initialized", extra={"level": config.debug.level})
    return logger

def main():
    try:
        config = AppConfig.load("config/default.yaml")
    except FileNotFoundError:
        print("Error: config/default.yaml not found.")
        sys.exit(1)

    logger = _setup_logging(config)

    logger.info("RCMAS - Pipeline Execution")
    logger.info("Grid: %sx%s | Agents: %s | Inaccessible: %s", config.grid.height, config.grid.width, config.agents.count, config.grid.inaccessible_sectors)

    context = PipelineContext(config=config)
    context.initialize_derived_values()

    logger.info("Derived: sectors=%s timesteps=%s", context.num_sectors, context.num_timesteps)

    max_episodes = config.simulation.max_episodes

    while context.iteration < max_episodes and not context.terminated:
        logger.info("Episode %s start", context.iteration + 1)
        context = run_episode(context)
        context.iteration += 1

    if context.ne_found:
        logger.info("Stopping early: Nash equilibrium detected.")
    elif context.terminated and not context.ne_found:
        logger.info("Stopping early: terminated without Nash equilibrium (e.g., baseline infeasible).")

if __name__ == "__main__":
    main()
