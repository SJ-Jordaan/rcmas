"""
Main RCMAS simulator implementing the complete simulation loop.
"""

from typing import List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from domain.agent import Agent
from domain.territory import Territory
from domain.state import State, FailureState
from domain.action import ActionProfile, ActionProfileBuilder
from simulation.protocol import ActionAvailabilityProtocol
from simulation.evolution import Evolution


class TerminationReason(Enum):
    """Reasons why a simulation terminated."""
    COLLISION = "collision"
    FULLY_OCCUPIED = "fully_occupied"
    MAX_ROUNDS_REACHED = "max_rounds_reached"


@dataclass
class SimulationStep:
    """Represents a single step in the simulation."""
    round_number: int
    state: State
    action_profile: Optional[ActionProfile]
    next_state: Optional[State]


@dataclass
class SimulationResult:
    """Complete result of a simulation run."""
    initial_state: State
    final_state: State
    trajectory: List[SimulationStep]
    termination_reason: TerminationReason
    total_rounds: int
    success: bool  # True if mission succeeded (no collision)
    agent_region_sizes: dict[Agent, int] = field(default_factory=dict)
    objectives_met: dict[Agent, bool] = field(default_factory=dict)


class RCMASSimulator:
    """
    Main simulator for Region Control Multi-Agent Systems.
    
    Orchestrates the complete simulation loop:
    1. Initialize state
    2. For each round:
       a. Get available actions for each agent
       b. Each agent selects an action via their strategy
       c. Create action profile
       d. Apply evolution to get next state
       e. Check for termination
    3. Compute final results and metrics
    """
    
    def __init__(
        self,
        territory: Territory,
        agents: List[Agent],
        max_rounds: Optional[int] = None
    ):
        """
        Initialize the simulator.
        
        Args:
            territory: The territory for the simulation
            agents: List of agents participating
            max_rounds: Optional maximum number of rounds (safety limit)
        """
        self.territory = territory
        self.agents = agents
        self.max_rounds = max_rounds or territory.size() * 2
        
        # Validate all agents have strategies
        for agent in agents:
            if not agent.has_strategy():
                raise ValueError(f"Agent {agent.id} does not have a strategy assigned")
        
        self._protocol = ActionAvailabilityProtocol()
        self._evolution = Evolution()
    
    def run(self, initial_state: Optional[State] = None) -> SimulationResult:
        """
        Run the complete simulation.
        
        Args:
            initial_state: Optional initial state (defaults to empty territory)
            
        Returns:
            Complete simulation result with trajectory and metrics
        """
        # Initialize
        if initial_state is None:
            state = State.create_initial_state(self.territory)
        else:
            state = initial_state.copy()
        
        initial_state_copy = state.copy()
        trajectory: List[SimulationStep] = []
        round_number = 0
        
        # Notify strategies that simulation is starting
        for agent in self.agents:
            agent.strategy.on_simulation_start(state, agent)
        
        # Main simulation loop
        while not self._evolution.is_terminal(state) and round_number < self.max_rounds:
            round_number += 1
            
            # Get available actions for each agent
            available_actions = self._protocol.get_all_available_actions(
                state, set(self.agents)
            )
            
            # Each agent selects an action
            builder = ActionProfileBuilder()
            for agent in self.agents:
                action = agent.strategy.select_action(
                    state, agent, available_actions[agent]
                )
                builder.add_action(agent, action)
            
            action_profile = builder.build()
            
            # Apply evolution
            next_state = self._evolution.apply(state, action_profile)
            
            # Record step
            trajectory.append(SimulationStep(
                round_number=round_number,
                state=state,
                action_profile=action_profile,
                next_state=next_state
            ))
            
            # Notify strategies that round is complete
            for agent in self.agents:
                agent.strategy.on_round_complete(
                    state, action_profile.get_action(agent), next_state, agent
                )
            
            # Transition to next state
            state = next_state
        
        # Determine termination reason
        if isinstance(state, FailureState):
            termination_reason = TerminationReason.COLLISION
            success = False
        elif state.is_fully_occupied():
            termination_reason = TerminationReason.FULLY_OCCUPIED
            success = True
        else:
            termination_reason = TerminationReason.MAX_ROUNDS_REACHED
            success = False
        
        # Compute final metrics (will implement region analysis next)
        agent_region_sizes = {}
        objectives_met = {}
        
        # For now, just use occupied sector counts as placeholder
        # (will be replaced with actual cohesive region computation)
        if not isinstance(state, FailureState):
            from analysis.regions import CohesiveRegionAnalyzer
            analyzer = CohesiveRegionAnalyzer(state)
            
            for agent in self.agents:
                max_region = analyzer.get_largest_cohesive_region(agent)
                region_size = len(max_region) if max_region else 0
                agent_region_sizes[agent] = region_size
                objectives_met[agent] = region_size >= agent.objective
        
        # Notify strategies that simulation has ended
        for agent in self.agents:
            agent.strategy.on_simulation_end(state, agent, success)
        
        return SimulationResult(
            initial_state=initial_state_copy,
            final_state=state,
            trajectory=trajectory,
            termination_reason=termination_reason,
            total_rounds=round_number,
            success=success,
            agent_region_sizes=agent_region_sizes,
            objectives_met=objectives_met
        )
    
    def run_multiple(
        self,
        num_runs: int,
        initial_state: Optional[State] = None
    ) -> List[SimulationResult]:
        """
        Run multiple independent simulations.
        
        Useful for stochastic strategies that may produce different outcomes.
        
        Args:
            num_runs: Number of simulations to run
            initial_state: Optional initial state for all runs
            
        Returns:
            List of simulation results
        """
        results = []
        for _ in range(num_runs):
            result = self.run(initial_state)
            results.append(result)
        return results
