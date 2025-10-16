"""
Basic example demonstrating RCMAS simulation.
"""

from domain import Territory, Agent, State
from strategies import RandomStrategy, GreedyExpansionStrategy
from simulation import RCMASSimulator
from visualization import TerminalVisualizer
from analysis import RegionMetrics


def main():
    """Run a basic RCMAS simulation."""
    
    print("=" * 60)
    print("RCMAS Simulation - Basic Example")
    print("=" * 60)
    print()
    
    # Create a 5x5 territory
    territory = Territory(width=5, height=5)
    print(f"Territory: {territory}")
    print()
    
    # Create two agents with different strategies
    agent1 = Agent(id="a1", objective=3, strategy=RandomStrategy(seed=42))
    agent2 = Agent(id="a2", objective=3, strategy=GreedyExpansionStrategy(seed=42))
    
    agents = [agent1, agent2]
    
    print("Agents:")
    for agent in agents:
        print(f"  {agent} with strategy {agent.strategy}")
    print()
    
    # Create simulator
    simulator = RCMASSimulator(territory, agents, max_rounds=50)
    
    # Run simulation
    print("Running simulation...")
    result = simulator.run()
    print()
    
    # Print results
    print("=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Termination: {result.termination_reason.value}")
    print(f"Total rounds: {result.total_rounds}")
    print(f"Success: {result.success}")
    print()
    
    # Visualize final state
    visualizer = TerminalVisualizer(color_enabled=True)
    print("Final State:")
    visualizer.print_state(result.final_state)
    print()
    
    # Show metrics
    if result.success:
        print("Agent Metrics:")
        for agent in agents:
            region_size = result.agent_region_sizes.get(agent, 0)
            objective_met = result.objectives_met.get(agent, False)
            print(f"  {agent.id}:")
            print(f"    Largest region: {region_size}")
            print(f"    Objective ({agent.objective}): {'✓ MET' if objective_met else '✗ NOT MET'}")
        print()
        
        # Determine winner
        winner = RegionMetrics.get_winner(result.final_state, agents)
        if winner:
            print(f"Winner: Agent {winner.id}")
        else:
            print("Result: TIE")
    else:
        print("Mission FAILED due to collision!")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
