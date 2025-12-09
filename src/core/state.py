# src/core/state.py
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional, Tuple
from .config import AppConfig
from enum import Enum


class PipelineMode(str, Enum):
  EVAL_BASELINE = "eval_baseline"
  AGENT_OPTIMIZATION = "agent_opt"


class PipelineContext(BaseModel):
  config: AppConfig
  num_sectors: int = 0
  num_timesteps: int = 0

  z3_optimizer: Any = None
  z3_model: Any = None
  z3_vars: Dict[str, Any] = Field(default_factory=dict)
  found_models: List[Any] = Field(default_factory=list)
  is_satisfiable: bool = False

  iteration: int = 0
  mode: PipelineMode = PipelineMode.EVAL_BASELINE
  target_agent_idx: Optional[int] = None
  current_baseline_payoff: float = 0.0
  terminated: bool = False

  # NEW STRUCTURE: Joint Q-Table
  # Key: State Tuple (s0, s1, ... sN)
  # Value: Dict[JointActionTuple, QValue]
  #        JointActionTuple = (action_agent_0, action_agent_1, ...)
  q_table: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = Field(default_factory=dict)

  # Debug/trace holders
  last_path: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = Field(default_factory=list)
  last_payoff: float = 0.0
  last_q_changes: List[Dict[str, Any]] = Field(default_factory=list)
  ne_found: bool = False

  class Config:
    arbitrary_types_allowed = True

  def initialize_derived_values(self):
    self.num_sectors = self.config.grid.height * self.config.grid.width
    if self.config.simulation.timesteps:
      self.num_timesteps = self.config.simulation.timesteps
    else:
      self.num_timesteps = self.num_sectors // self.config.agents.count
