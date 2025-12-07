from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
from .config import AppConfig


class PipelineContext(BaseModel):
  # Static Configuration
  config: AppConfig

  # Derived Constants
  num_sectors: int = 0
  num_timesteps: int = 0

  # Solver State
  z3_optimizer: Any = None
  z3_model: Any = None

  # --- MISSING FIELDS ADDED HERE ---
  z3_vars: Dict[str, Any] = Field(default_factory=dict)  # Stores 'state' and 'action' lists
  found_models: List[Any] = Field(default_factory=list)  # Stores all found Z3 models
  # ---------------------------------

  is_satisfiable: bool = False

  # RL State
  iteration: int = 0
  current_reward: float = 0.0
  terminated: bool = False

  class Config:
    arbitrary_types_allowed = True

  def initialize_derived_values(self):
    self.num_sectors = self.config.grid.height * self.config.grid.width
    if self.config.simulation.timesteps:
      self.num_timesteps = self.config.simulation.timesteps
    else:
      self.num_timesteps = self.num_sectors // self.config.agents.count
