"""
Q-Learning training example for RCMAS.

This example demonstrates:
1. Training Q-Learning agents over multiple episodes
2. Tracking learning progress
3. Comparing trained vs untrained performance
"""

from domain import Territory, Agent
from strategies import QLearningStrategy, RandomStrategy, GreedyExpansionStrategy
from simulation import RCMASSimulator
from visualization import TerminalVisualizer
from analysis import RegionMetrics


def train_qlearning_agent(territory, agent, opponent, num_episodes=100, verbose=False):
    """
    Train a Q-Learning agent over multiple episodes.
    
    Args:
        territory: The territory to train on
        agent: The Q-Learning agent to train
        opponent: The opponent agent
        num_episodes: Number of training episodes
        verbose: Whether to print progress
        
    Returns:
        List of success rates over time
    """
    success_count = 0
    success_rates = []
    
    for episode in range(num_episodes):
        # Run simulation
        simulator = RCMASSimulator(territory, [agent, opponent], max_rounds=50)
        result = simulator.run()
        
        # Track success
        if result.success:
            success_count += 1
        
        # Calculate success rate (over last 10 episodes)
        if (episode + 1) % 10 == 0:
            success_rate = success_count / 10
            success_rates.append(success_rate)
            
            if verbose:
                epsilon_str = f"ε={agent.strategy.epsilon:.3f}"
                print(f"Episode {episode + 1}: Success rate = {success_rate:.2%}, {epsilon_str}")
            
            success_count = 0
    
    return success_rates


def main():
    """Run Q-Learning training example."""
    
    print("=" * 70)
    print("Q-LEARNING TRAINING FOR RCMAS")
    print("=" * 70)
    print()
    
    # Create a small territory for faster training
    territory = Territory(width=4, height=4)
    print(f"Territory: {territory}")
    print()
    
    # Create Q-Learning agent
    q_agent = Agent(
        id="q1",
        objective=3,
        strategy=QLearningStrategy(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon_start=1.0,  # Start with full exploration
            epsilon_end=0.01,   # Decay to 1% exploration
            epsilon_decay=0.995, # Slower decay for better convergence
            seed=42
        )
    )
    
    # Create random opponent
    random_opponent = Agent(
        id="r1",
        objective=3,
        strategy=GreedyExpansionStrategy(seed=99)
    )
    
    print("Training Configuration:")
    print(f"  Q-Learning Agent: {q_agent.id}")
    print(f"  Learning Rate (α): {q_agent.strategy.alpha}")
    print(f"  Discount Factor (γ): {q_agent.strategy.gamma}")
    print(f"  Exploration Rate (ε): {q_agent.strategy.epsilon:.3f} → {q_agent.strategy.epsilon_end}")
    print(f"  Exploration Decay: {q_agent.strategy.epsilon_decay}")
    print(f"  Opponent: Greedy expansion strategy")
    print()
    
    # Train the agent
    print("=" * 70)
    print("TRAINING PHASE (200 episodes)")
    print("=" * 70)
    
    success_rates = train_qlearning_agent(
        territory, q_agent, random_opponent,
        num_episodes=20000,
        verbose=True
    )
    
    print()
    print("Training complete!")
    print(f"Q-table size: {len(q_agent.strategy.q_table)} state-action pairs learned")
    print(f"Final exploration rate (ε): {q_agent.strategy.epsilon:.4f}")
    print()
    
    # Switch to exploitation mode
    print("=" * 70)
    print("EVALUATION PHASE (exploitation mode, ε=0)")
    print("=" * 70)
    q_agent.strategy.epsilon = 0.0  # Pure exploitation
    q_agent.strategy.set_training_mode(False)
    
    # Run evaluation episodes
    eval_episodes = 10
    successes = 0
    total_rounds = []
    agent_scores = []
    
    for i in range(eval_episodes):
        simulator = RCMASSimulator(territory, [q_agent, random_opponent], max_rounds=50)
        result = simulator.run()
        
        if result.success:
            successes += 1
            q_score = result.agent_region_sizes.get(q_agent, 0)
            agent_scores.append(q_score)
        
        total_rounds.append(result.total_rounds)
        
        print(f"Eval {i+1}: {'SUCCESS' if result.success else 'FAILED (collision)'} "
              f"- Rounds: {result.total_rounds}")
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Success Rate: {successes}/{eval_episodes} = {successes/eval_episodes:.1%}")
    
    if agent_scores:
        print(f"Average Q-Agent Score: {sum(agent_scores)/len(agent_scores):.1f} sectors")
        print(f"Q-Agent Objective: {q_agent.objective} sectors")
    
    print(f"Average Rounds: {sum(total_rounds)/len(total_rounds):.1f}")
    print()
    
    # Run one final demo simulation with visualization
    print("=" * 70)
    print("DEMO SIMULATION (with visualization)")
    print("=" * 70)
    
    simulator = RCMASSimulator(territory, [q_agent, random_opponent], max_rounds=50)
    result = simulator.run()
    
    # Visualize
    viz = TerminalVisualizer(color_enabled=True)
    viz.print_state(result.final_state)
    print()
    
    print(f"Outcome: {'SUCCESS' if result.success else 'COLLISION'}")
    print(f"Rounds: {result.total_rounds}")
    
    if result.success:
        print()
        print("Agent Performance:")
        for agent in [q_agent, random_opponent]:
            region_size = result.agent_region_sizes.get(agent, 0)
            objective_met = result.objectives_met.get(agent, False)
            print(f"  {agent.id}: Region={region_size}, "
                  f"Objective={agent.objective}, "
                  f"Met={'✓' if objective_met else '✗'}")
        
        winner = RegionMetrics.get_winner(result.final_state, [q_agent, random_opponent])
        if winner:
            print(f"\nWinner: {winner.id}")
    
    print()
    print("=" * 70)
    
    # Learning progress analysis
    print()
    print("LEARNING PROGRESS:")
    print("(Success rate over time in 10-episode windows)")
    for i, rate in enumerate(success_rates):
        episodes = f"{i*10+1}-{(i+1)*10}"
        bar = "█" * int(rate * 40)
        print(f"  Episodes {episodes:>7}: {bar} {rate:.1%}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
