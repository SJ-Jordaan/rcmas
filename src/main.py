# src/main.py
import sys
from src.core.config import AppConfig
from src.core.state import PipelineContext
from src.handlers.encoder import EncodingHandler
from src.handlers.solver import SolverHandler
from src.handlers.output import OutputHandler
from src.handlers.learner import LearningHandler

def main():
    print("=" * 60)
    print("RCMAS - Pipeline Execution")
    print("=" * 60)

    try:
        config = AppConfig.load("config/default.yaml")
    except FileNotFoundError:
        print("Error: config/default.yaml not found.")
        sys.exit(1)

    context = PipelineContext(config=config)
    context.initialize_derived_values()

    print(f"Grid: {config.grid.height}x{config.grid.width}")
    print(f"Agents: {config.agents.count}")
    print(f"Sectors: {context.num_sectors}")
    print(f"Timesteps: {context.num_timesteps}")
    print("=" * 60)

    pipeline = [
        EncodingHandler(),
        SolverHandler(),
        OutputHandler(),
        LearningHandler()
    ]

    MAX_EPISODES = 5  # Could be moved to config

    while context.iteration < MAX_EPISODES:
        print(f"\n--- Episode {context.iteration + 1} ---")

        # Run each handler in sequence
        for handler in pipeline:
            context = handler.handle(context)

            # If the solver failed entirely, we might want to break the pipeline early
            # but usually, the Learner needs to see the failure to penalize it.
            if isinstance(handler, SolverHandler) and not context.is_satisfiable:
                print("Solver failed to find any model this episode.")

        # Check termination condition from RL agent
        if context.terminated:
            print("Goal met. Terminating pipeline.")
            break

        context.iteration += 1

if __name__ == "__main__":
    main()
