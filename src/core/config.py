from pydantic import BaseModel
from typing import List, Optional
import yaml

class GridConfig(BaseModel):
    height: int
    width: int
    inaccessible_sectors: List[int] = []

class AgentConfig(BaseModel):
    count: int

# --- NEW: Defined to match the 'simulation' block in YAML ---
class SimulationConfig(BaseModel):
    timesteps: Optional[int] = None
    max_models: int = 1  # Default to 1 if not in YAML
# ------------------------------------------------------------

class AppConfig(BaseModel):
    grid: GridConfig
    agents: AgentConfig
    simulation: SimulationConfig  # <--- This field was missing!

    @classmethod
    def load(cls, path: str = "config/default.yaml") -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
