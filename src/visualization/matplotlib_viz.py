"""
Matplotlib-based visualization for RCMAS.
"""

from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np

from domain.state import State
from domain.agent import Agent, DUMMY_AGENT
from domain.territory import Territory, Sector


class MatplotlibVisualizer:
    """
    Matplotlib-based visualization for RCMAS states.
    
    Creates publication-quality figures for papers and presentations.
    """
    
    def __init__(self, figsize: tuple = (10, 10)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self._agent_colors: Dict[str, str] = {}
        self._color_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', 
                               '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        self._next_color_idx = 0
    
    def visualize_state(
        self,
        state: State,
        title: Optional[str] = None,
        show_grid: bool = True,
        show_legend: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create a matplotlib figure of a state.
        
        Args:
            state: The state to visualize
            title: Optional title
            show_grid: Whether to show grid lines
            show_legend: Whether to show legend
            ax: Optional axes to plot on (creates new figure if None)
            
        Returns:
            Figure object
        """
        territory = state.get_territory()
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Create grid data
        grid_data = np.zeros((territory.height, territory.width))
        
        # Build agent to color index mapping
        agents_present = set()
        for i in range(1, territory.height + 1):
            for j in range(1, territory.width + 1):
                sector = Sector(i, j)
                if territory.contains(sector):
                    occupant = state.get_occupant(sector)
                    if occupant != DUMMY_AGENT:
                        agents_present.add(occupant)
        
        agent_to_idx = {agent: idx + 1 for idx, agent in enumerate(sorted(agents_present, key=lambda a: a.id))}
        
        # Fill grid
        for i in range(1, territory.height + 1):
            for j in range(1, territory.width + 1):
                sector = Sector(i, j)
                
                if not territory.contains(sector):
                    grid_data[i-1, j-1] = -1  # Inaccessible
                else:
                    occupant = state.get_occupant(sector)
                    if occupant == DUMMY_AGENT:
                        grid_data[i-1, j-1] = 0  # Unoccupied
                    else:
                        grid_data[i-1, j-1] = agent_to_idx[occupant]
        
        # Create colormap
        colors = ['#ecf0f1']  # Unoccupied (light gray)
        for agent in sorted(agents_present, key=lambda a: a.id):
            colors.append(self._get_agent_color(agent))
        
        cmap = ListedColormap(['#34495e'] + colors)  # Dark gray for inaccessible
        
        # Plot
        im = ax.imshow(grid_data, cmap=cmap, vmin=-1, vmax=len(agents_present),
                      aspect='equal', interpolation='nearest')
        
        # Grid
        if show_grid:
            ax.set_xticks(np.arange(-0.5, territory.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, territory.height, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Labels
        ax.set_xticks(range(territory.width))
        ax.set_yticks(range(territory.height))
        ax.set_xticklabels(range(1, territory.width + 1))
        ax.set_yticklabels(range(1, territory.height + 1))
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        if show_legend and agents_present:
            legend_elements = []
            for agent in sorted(agents_present, key=lambda a: a.id):
                color = self._get_agent_color(agent)
                legend_elements.append(
                    patches.Patch(facecolor=color, edgecolor='black',
                                 label=f'Agent {agent.id} (obj={agent.objective})')
                )
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        return fig
    
    def visualize_trajectory(
        self,
        states: List[State],
        titles: Optional[List[str]] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None
    ) -> plt.Figure:
        """
        Visualize multiple states in a grid.
        
        Args:
            states: List of states
            titles: Optional list of titles
            rows: Number of rows (auto-calculated if None)
            cols: Number of columns (auto-calculated if None)
            
        Returns:
            Figure with subplots
        """
        n_states = len(states)
        
        # Calculate grid dimensions
        if rows is None and cols is None:
            cols = int(np.ceil(np.sqrt(n_states)))
            rows = int(np.ceil(n_states / cols))
        elif rows is None:
            rows = int(np.ceil(n_states / cols))
        elif cols is None:
            cols = int(np.ceil(n_states / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        
        # Flatten axes array for easier indexing
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for idx, state in enumerate(states):
            title = titles[idx] if titles and idx < len(titles) else f"Round {idx}"
            self.visualize_state(state, title=title, show_legend=False, ax=axes[idx])
        
        # Hide unused subplots
        for idx in range(n_states, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Figure to save
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    def _get_agent_color(self, agent: Agent) -> str:
        """Get consistent color for an agent."""
        if agent.id not in self._agent_colors:
            if self._next_color_idx < len(self._color_palette):
                self._agent_colors[agent.id] = self._color_palette[self._next_color_idx]
                self._next_color_idx += 1
            else:
                # Generate random color
                import random
                self._agent_colors[agent.id] = f'#{random.randint(0, 0xFFFFFF):06x}'
        
        return self._agent_colors[agent.id]
