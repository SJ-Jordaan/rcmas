"""
Q-Learning based strategy for RCMAS (placeholder for future RL integration).
"""

from typing import Set, Dict, Tuple, Optional
import random

from domain.state import State
from domain.agent import Agent
from domain.action import Action
from strategies.abstract_strategy import AbstractStrategy


class QLearningStrategy(AbstractStrategy):
    """
    Q-Learning strategy for RCMAS agents.
    
    This is a placeholder implementation that demonstrates the integration
    point for reinforcement learning. Future work will integrate with
    stable-baselines3 or custom RL implementations.
    
    Q-Learning learns a Q-value function Q(s, a) representing the expected
    cumulative reward of taking action a in state s.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Initialize Q-Learning strategy.
        
        Args:
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon_start: Initial exploration rate for epsilon-greedy policy
            epsilon_end: Minimum exploration rate (after decay)
            epsilon_decay: Multiplicative decay factor applied after each episode
            seed: Random seed for reproducibility
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Q-table: maps (state_hash, action) -> Q-value
        # Note: In practice, state representation needs careful design
        self.q_table: Dict[Tuple[int, Action], float] = {}
        
        # Training mode flag
        self.training = True
        
        # Episode memory for learning
        self.episode_history: list = []
    
    def select_action(
        self,
        state: State,
        agent: Agent,
        available_actions: Set[Action]
    ) -> Action:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            agent: The agent making the decision
            available_actions: Set of available actions
            
        Returns:
            Selected action
        """
        if not available_actions:
            raise ValueError(f"No available actions for agent {agent.id}")
        
        # Epsilon-greedy exploration
        if self.training and self.rng.random() < self.epsilon:
            # Explore: random action
            action = self.rng.choice(list(available_actions))
        else:
            # Exploit: best action according to Q-values
            action = self._get_best_action(state, available_actions)
        
        # Store state-action pair for learning
        if self.training:
            self.episode_history.append((hash(state), action))
        
        return action
    
    def _get_best_action(self, state: State, available_actions: Set[Action]) -> Action:
        """
        Get the action with highest Q-value.
        
        Args:
            state: Current state
            available_actions: Available actions
            
        Returns:
            Action with highest Q-value
        """
        state_hash = hash(state)
        best_action = None
        best_q_value = float('-inf')
        
        for action in available_actions:
            q_value = self.q_table.get((state_hash, action), 0.0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        # If all Q-values are equal (e.g., unseen state), choose randomly
        if best_action is None or best_q_value == float('-inf'):
            best_action = self.rng.choice(list(available_actions))
        
        return best_action
    
    def _get_reward(
        self,
        old_state: State,
        action: Action,
        new_state: State,
        agent: Agent
    ) -> float:
        """
        Compute reward for a state transition.
        
        Enhanced reward structure:
        - Region growth: Reward for increasing cohesive region size
        - Compactness: Bonus for maintaining compact regions (avoid fragmentation)
        - Strategic positioning: Reward for occupying central territory sectors
        - Defensive play: Bonus for blocking opponent expansion routes
        - Objective achievement: Large bonus for reaching objective
        - Efficiency: Higher rewards for faster completion
        
        Args:
            old_state: State before action
            action: Action taken
            new_state: State after action
            agent: The agent
            
        Returns:
            Reward value
        """
        from analysis.regions import CohesiveRegionAnalyzer
        
        reward = 0.0
        
        # Calculate region size change
        old_analyzer = CohesiveRegionAnalyzer(old_state)
        new_analyzer = CohesiveRegionAnalyzer(new_state)
        
        old_max_region = old_analyzer.get_largest_cohesive_region(agent)
        new_max_region = new_analyzer.get_largest_cohesive_region(agent)
        
        old_size = len(old_max_region) if old_max_region else 0
        new_size = len(new_max_region) if new_max_region else 0
        
        # 1. REGION GROWTH: Scale up importance (was just the delta)
        region_delta = new_size - old_size
        reward += region_delta * 5.0
        
        # 2. COMPACTNESS: Reward compact regions (avoid fragmentation)
        if new_size > 0 and new_max_region:
            new_perimeter = new_analyzer.get_perimeter_sectors(new_max_region)
            perimeter_size = len(new_perimeter)
            if perimeter_size > 0:
                # Compactness ratio: higher is better (circle = 1, line = 0)
                # A perfect square of size 9 has perimeter 12, ratio = 9/12 = 0.75
                compactness = new_size / perimeter_size
                reward += compactness * 2.0
        
        # 3. STRATEGIC POSITIONING: Reward central sectors
        # Central sectors provide better expansion opportunities
        territory = old_state.get_territory()
        center_i = territory.height / 2.0
        center_j = territory.width / 2.0
        target = action.sector
        
        # Distance from center (normalized)
        di = abs(target.i - center_i) / territory.height
        dj = abs(target.j - center_j) / territory.width
        centrality = 1.0 - ((di + dj) / 2.0)  # 1.0 = center, 0.0 = corner
        reward += centrality * 1.5
        
        # 4. DEFENSIVE PLAY: Block opponent expansion
        # Check if we occupied a sector adjacent to opponent territory
        occupancy_counts = old_state.get_agent_occupancy_counts()
        opponent_agents = [a for a in occupancy_counts.keys() if a != agent]
        if opponent_agents:
            opponent = opponent_agents[0]  # Assume 2-player for now
            opponent_sectors = old_state.get_sectors_for_agent(opponent)
            
            # Check if target sector is adjacent to opponent
            for opp_sector in opponent_sectors:
                if target in opp_sector.get_adjacent_sectors():
                    reward += 3.0  # Blocking opponent is valuable
                    break
        
        # 5. OBJECTIVE ACHIEVEMENT: Large bonus for reaching objective
        if new_size >= agent.objective and old_size < agent.objective:
            reward += 20.0
        
        return reward
    
    def on_round_complete(
        self,
        old_state: State,
        action: Action,
        new_state: State,
        agent: Agent
    ) -> None:
        """
        Update Q-values based on the observed transition.
        
        Args:
            old_state: State before action
            action: Action taken
            new_state: State after action
            agent: The agent
        """
        if not self.training:
            return
        
        old_state_hash = hash(old_state)
        new_state_hash = hash(new_state)
        
        # Get reward
        reward = self._get_reward(old_state, action, new_state, agent)
        
        # Get current Q-value
        current_q = self.q_table.get((old_state_hash, action), 0.0)
        
        # Get max Q-value for next state (will be 0 if terminal)
        if new_state.is_failure() or new_state.is_fully_occupied():
            max_next_q = 0.0
        else:
            # In a real implementation, we'd need to know available actions
            # For now, use 0 (conservative estimate)
            max_next_q = 0.0
        
        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(old_state_hash, action)] = new_q
    
    def on_simulation_end(
        self,
        final_state: State,
        agent: Agent,
        success: bool
    ) -> None:
        """
        Apply final reward/penalty at end of episode.
        
        Args:
            final_state: Final state reached
            agent: The agent
            success: Whether mission succeeded
        """
        if not self.training:
            return
        
        # Apply terminal reward
        if not success:
            # Large penalty for collision
            terminal_reward = -100.0
        else:
            # Reward based on final region size vs objective
            from analysis.regions import CohesiveRegionAnalyzer
            analyzer = CohesiveRegionAnalyzer(final_state)
            max_region = analyzer.get_largest_cohesive_region(agent)
            region_size = len(max_region) if max_region else 0
            
            if region_size >= agent.objective:
                terminal_reward = 50.0
            else:
                terminal_reward = -10.0
        
        # Apply terminal reward to last state-action pair
        if self.episode_history:
            last_state_hash, last_action = self.episode_history[-1]
            current_q = self.q_table.get((last_state_hash, last_action), 0.0)
            new_q = current_q + self.alpha * terminal_reward
            self.q_table[(last_state_hash, last_action)] = new_q
        
        # Clear episode history
        self.episode_history.clear()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def set_training_mode(self, training: bool) -> None:
        """Enable or disable training mode."""
        self.training = training
    
    def get_name(self) -> str:
        return "Q-Learning"
    
    def __repr__(self) -> str:
        return f"QLearningStrategy(α={self.alpha}, γ={self.gamma}, ε={self.epsilon})"
