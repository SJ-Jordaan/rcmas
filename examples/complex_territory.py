"""
Example with obstacles and complex territory.
"""

from domain import TerritoryBuilder, Agent
from strategies import GreedyExpansionStrategy, RandomStrategy
from simulation import RCMASSimulator
from visualization import TerminalVisualizer, MatplotlibVisualizer
from analysis import CohesiveRegionAnalyzer


def main():
    """Run simulation with obstacles."""
    
    print("=" * 60)
    print("RCMAS Simulation - Complex Territory with Obstacles")
    print("=" * 60)
    print()
    
    # Build a territory with obstacles
    territory = (TerritoryBuilder(10, 10)
                 .add_obstacle_region(4, 4, 4, 7)  # Horizontal barrier
                 .add_obstacle_region(6, 4, 6, 7)  # Another barrier
                 .add_obstacle(5, 5)                # Single obstacle
                 .build())
    
    print(f"Territory: {territory}")
    print()
    
    # Create three agents
    agent1 = Agent(id="a1", objective=5, strategy=GreedyExpansionStrategy(seed=1))
    agent2 = Agent(id="a2", objective=5, strategy=GreedyExpansionStrategy(seed=2))
    agent3 = Agent(id="a3", objective=4, strategy=RandomStrategy(seed=3))
    
    agents = [agent1, agent2, agent3]
    
    # Visualize initial state
    initial_state = territory
    from domain import State
    state = State.create_initial_state(territory)
    
    terminal_viz = TerminalVisualizer(color_enabled=True)
    print("Initial Territory (with obstacles):")
    terminal_viz.print_state(state)
    print()
    
    # Run simulation
    simulator = RCMASSimulator(territory, agents, max_rounds=100)
    print("Running simulation...")
    result = simulator.run()
    print()
    
    # Results
    print("=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Termination: {result.termination_reason.value}")
    print(f"Total rounds: {result.total_rounds}")
    print(f"Success: {result.success}")
    print()
    
    # Final state
    print("Final State:")
    terminal_viz.print_state(result.final_state)
    print()
    
    # Detailed metrics
    if result.success:
        print("Detailed Agent Analysis:")
        analyzer = CohesiveRegionAnalyzer(result.final_state)
        
        for agent in agents:
            regions = analyzer.get_all_cohesive_regions(agent)
            region_sizes = analyzer.get_region_sizes(agent)
            max_size = analyzer.get_max_region_size(agent)
            fragmentation = analyzer.get_fragmentation_score(agent)
            
            print(f"\n  Agent {agent.id}:")
            print(f"    Objective: {agent.objective}")
            print(f"    Number of regions: {len(regions)}")
            print(f"    Region sizes: {region_sizes}")
            print(f"    Largest region: {max_size}")
            print(f"    Fragmentation: {fragmentation:.3f}")
            print(f"    Objective met: {'✓ YES' if max_size >= agent.objective else '✗ NO'}")
        
        print()
        
        # Try to create matplotlib visualization
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            mpl_viz = MatplotlibVisualizer(figsize=(8, 8))
            fig = mpl_viz.visualize_state(result.final_state, 
                                         title="Final Territory State")
            mpl_viz.save_figure(fig, "complex_territory_result.png")
            print("Visualization saved to: complex_territory_result.png")
        except ImportError:
            print("(Matplotlib not available for saving figure)")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
