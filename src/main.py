# src/main.py
import sys
from src.core.config import AppConfig
from src.core.state import PipelineContext
from src.handlers.encoder import SATEncodingHandler
from src.handlers.solver import Z3SolverHandler
from src.handlers.output import OutputHandler
from src.handlers.learner import QLearningHandler

def main():
    print("=" * 60)
    print("RCMAS - Pipeline Execution")
    print("=" * 60)

    # 1. Load Configuration
    try:
        config = AppConfig.load("config/default.yaml")
    except FileNotFoundError:
        print("Error: config/default.yaml not found.")
        sys.exit(1)

    # 2. Initialize Shared Context
    context = PipelineContext(config=config)
    context.initialize_derived_values()

    print(f"Grid: {config.grid.height}x{config.grid.width}")
    print(f"Agents: {config.agents.count}")
    print(f"Sectors: {context.num_sectors}")
    print(f"Timesteps: {context.num_timesteps}")
    print("=" * 60)

    # 3. Define the Pipeline Stages
    # The order matters: Encode -> Solve (Loop) -> Output/Visualize -> Learn -> Repeat
    pipeline = [
        SATEncodingHandler(),
        Z3SolverHandler(),
        OutputHandler(),
        QLearningHandler()
    ]

    # 4. Main Execution Loop (The RL Episode Loop)
    MAX_EPISODES = 5  # Could be moved to config

    while context.iteration < MAX_EPISODES:
        print(f"\n--- Episode {context.iteration + 1} ---")

        # Run each handler in sequence
        for handler in pipeline:
            context = handler.handle(context)

            # If the solver failed entirely, we might want to break the pipeline early
            # but usually, the Learner needs to see the failure to penalize it.
            if isinstance(handler, Z3SolverHandler) and not context.is_satisfiable:
                print("Solver failed to find any model this episode.")

        # Check termination condition from RL agent
        if context.terminated:
            print("Goal met. Terminating pipeline.")
            break

        context.iteration += 1

if __name__ == "__main__":
    main()
