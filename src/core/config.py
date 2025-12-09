from pydantic import BaseModel
from typing import List, Optional
import yaml

class GridConfig(BaseModel):
    height: int
    width: int
    inaccessible_sectors: List[int] = []


class AgentConfig(BaseModel):
    count: int


class SimulationConfig(BaseModel):
    timesteps: Optional[int] = None
    max_models: int = 1
    max_episodes: int = 50


class DebugConfig(BaseModel):
    enabled: bool = True
    level: str = "INFO"
    max_q_deltas: int = 20
    max_state_logs: int = 10
    dump_q_json: bool = False
    epsilon: float = 1e-6
    include_strategy_constraints: bool = True
    dump_strategy_constraints: bool = False
    dump_constraints: bool = False
    max_init_states: int = 500
    max_actions_per_state: int = 50
    learning_rate: float = 0.5
    discount: float = 0.9


class AppConfig(BaseModel):
    grid: GridConfig
    agents: AgentConfig
    simulation: SimulationConfig
    debug: DebugConfig = DebugConfig()

    @classmethod
    def load(cls, path: str = "config/default.yaml") -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
