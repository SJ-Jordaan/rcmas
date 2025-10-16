"""
Territory and Sector classes representing the spatial domain of RCMAS.
"""

from typing import Set, Tuple, Optional, Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sector:
    """
    Represents a single sector (i, j) in the territory.
    
    Immutable value object representing a position in the territory grid.
    """
    i: int  # Row coordinate
    j: int  # Column coordinate
    
    def __str__(self) -> str:
        return f"({self.i},{self.j})"
    
    def __repr__(self) -> str:
        return f"Sector({self.i}, {self.j})"
    
    def get_adjacent_sectors(self) -> list['Sector']:
        """
        Returns the four orthogonally adjacent sectors.
        
        Returns:
            List of adjacent sectors (up, down, left, right)
        """
        return [
            Sector(self.i + 1, self.j),  # Down
            Sector(self.i - 1, self.j),  # Up
            Sector(self.i, self.j + 1),  # Right
            Sector(self.i, self.j - 1),  # Left
        ]


class Territory:
    """
    Represents the territory T ⊆ {(i,j) | 1 ≤ i ≤ w, 1 ≤ j ≤ r}.
    
    The territory is a set of accessible sectors in a grid. Some sectors
    may be marked as inaccessible to model obstacles or boundaries.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        inaccessible_sectors: Optional[Set[Sector]] = None
    ):
        """
        Initialize a territory.
        
        Args:
            width: Width of the territory (w in the paper)
            height: Height of the territory (r in the paper)
            inaccessible_sectors: Set of sectors that are not part of the territory
        """
        if width <= 0 or height <= 0:
            raise ValueError("Territory dimensions must be positive")
        
        self._width = width
        self._height = height
        self._inaccessible_sectors = inaccessible_sectors or set()
        
        # Build the set of accessible sectors
        self._sectors: Set[Sector] = {
            Sector(i, j)
            for i in range(1, width + 1)
            for j in range(1, height + 1)
            if Sector(i, j) not in self._inaccessible_sectors
        }
    
    @property
    def width(self) -> int:
        """Returns the width of the territory."""
        return self._width
    
    @property
    def height(self) -> int:
        """Returns the height of the territory."""
        return self._height
    
    @property
    def sectors(self) -> Set[Sector]:
        """Returns the set of all accessible sectors."""
        return self._sectors.copy()
    
    def contains(self, sector: Sector) -> bool:
        """
        Check if a sector is part of the territory.
        
        Args:
            sector: The sector to check
            
        Returns:
            True if the sector is accessible, False otherwise
        """
        return sector in self._sectors
    
    def get_adjacent_accessible_sectors(self, sector: Sector) -> list[Sector]:
        """
        Get all adjacent sectors that are accessible.
        
        Args:
            sector: The sector whose neighbors to find
            
        Returns:
            List of accessible adjacent sectors
        """
        return [
            adj for adj in sector.get_adjacent_sectors()
            if self.contains(adj)
        ]
    
    def size(self) -> int:
        """Returns the total number of accessible sectors."""
        return len(self._sectors)
    
    def __iter__(self) -> Iterator[Sector]:
        """Iterate over all accessible sectors."""
        return iter(self._sectors)
    
    def __len__(self) -> int:
        """Returns the number of accessible sectors."""
        return len(self._sectors)
    
    def __str__(self) -> str:
        return f"Territory({self._width}x{self._height}, {len(self._sectors)} sectors)"
    
    def __repr__(self) -> str:
        return self.__str__()


class TerritoryBuilder:
    """
    Builder pattern for constructing complex territory configurations.
    """
    
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._inaccessible: Set[Sector] = set()
    
    def add_obstacle(self, i: int, j: int) -> 'TerritoryBuilder':
        """Add a single obstacle at position (i, j)."""
        self._inaccessible.add(Sector(i, j))
        return self
    
    def add_obstacle_region(
        self,
        start_i: int,
        start_j: int,
        end_i: int,
        end_j: int
    ) -> 'TerritoryBuilder':
        """Add a rectangular region of obstacles."""
        for i in range(start_i, end_i + 1):
            for j in range(start_j, end_j + 1):
                self._inaccessible.add(Sector(i, j))
        return self
    
    def add_border_obstacles(self, thickness: int = 1) -> 'TerritoryBuilder':
        """Add obstacles around the border of the territory."""
        for t in range(thickness):
            # Top and bottom borders
            for j in range(1, self._height + 1):
                self._inaccessible.add(Sector(1 + t, j))
                self._inaccessible.add(Sector(self._width - t, j))
            
            # Left and right borders
            for i in range(1, self._width + 1):
                self._inaccessible.add(Sector(i, 1 + t))
                self._inaccessible.add(Sector(i, self._height - t))
        
        return self
    
    def build(self) -> Territory:
        """Build and return the configured territory."""
        return Territory(self._width, self._height, self._inaccessible)
