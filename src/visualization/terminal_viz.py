"""
Terminal-based visualization for RCMAS states.
"""

from typing import Dict, List, Optional
from domain.state import State
from domain.agent import Agent, DUMMY_AGENT
from domain.territory import Territory, Sector


class TerminalVisualizer:
    """
    Simple terminal-based visualization of RCMAS states.
    
    Displays the territory grid with agent occupancy.
    """
    
    def __init__(self, color_enabled: bool = True):
        """
        Initialize visualizer.
        
        Args:
            color_enabled: Whether to use ANSI color codes
        """
        self.color_enabled = color_enabled
        self._agent_symbols: Dict[str, str] = {}
        self._next_symbol_idx = 0
        self._symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        
        # ANSI color codes
        self._colors = [
            '\033[91m',  # Red
            '\033[94m',  # Blue
            '\033[92m',  # Green
            '\033[93m',  # Yellow
            '\033[95m',  # Magenta
            '\033[96m',  # Cyan
        ]
        self._reset = '\033[0m'
    
    def visualize_state(self, state: State, title: Optional[str] = None) -> str:
        """
        Create a string visualization of a state.
        
        Args:
            state: The state to visualize
            title: Optional title to display above the visualization
            
        Returns:
            String representation of the state
        """
        territory = state.get_territory()
        lines = []
        
        if title:
            lines.append(title)
            lines.append("=" * len(title))
        
        # Build the grid
        for i in range(1, territory.height + 1):
            row = []
            for j in range(1, territory.width + 1):
                sector = Sector(i, j)
                
                if not territory.contains(sector):
                    # Inaccessible sector
                    row.append('█')
                else:
                    occupant = state.get_occupant(sector)
                    if occupant == DUMMY_AGENT:
                        row.append('·')
                    else:
                        symbol = self._get_agent_symbol(occupant)
                        if self.color_enabled:
                            color = self._get_agent_color(occupant)
                            row.append(f"{color}{symbol}{self._reset}")
                        else:
                            row.append(symbol)
            
            lines.append(' '.join(row))
        
        # Add legend
        lines.append("")
        lines.append("Legend:")
        for agent_id, symbol in sorted(self._agent_symbols.items()):
            lines.append(f"  {symbol} = Agent {agent_id}")
        lines.append("  · = Unoccupied")
        lines.append("  █ = Inaccessible")
        
        return '\n'.join(lines)
    
    def print_state(self, state: State, title: Optional[str] = None) -> None:
        """
        Print state visualization to stdout.
        
        Args:
            state: The state to visualize
            title: Optional title
        """
        print(self.visualize_state(state, title))
    
    def _get_agent_symbol(self, agent: Agent) -> str:
        """Get consistent symbol for an agent."""
        if agent.id not in self._agent_symbols:
            if self._next_symbol_idx < len(self._symbols):
                self._agent_symbols[agent.id] = self._symbols[self._next_symbol_idx]
                self._next_symbol_idx += 1
            else:
                # Fallback to using first letter of agent id
                self._agent_symbols[agent.id] = agent.id[0].upper()
        
        return self._agent_symbols[agent.id]
    
    def _get_agent_color(self, agent: Agent) -> str:
        """Get consistent color for an agent."""
        # Use hash of agent id to get consistent color
        color_idx = hash(agent.id) % len(self._colors)
        return self._colors[color_idx]
    
    def visualize_trajectory(
        self,
        states: List[State],
        round_labels: Optional[List[str]] = None
    ) -> str:
        """
        Visualize a sequence of states.
        
        Args:
            states: List of states to visualize
            round_labels: Optional labels for each state
            
        Returns:
            String with all visualizations
        """
        lines = []
        
        for idx, state in enumerate(states):
            if round_labels and idx < len(round_labels):
                title = round_labels[idx]
            else:
                title = f"Round {idx}"
            
            lines.append(self.visualize_state(state, title))
            lines.append("\n")
        
        return '\n'.join(lines)
