from abc import ABC, abstractmethod
from typing import List, Any
from src.core.state import PipelineContext


class BaseConstraint(ABC):
  """
  Abstract base class for all RCMAS constraints.
  """

  @abstractmethod
  def build(self, ctx: PipelineContext) -> List[Any]:
    """
    Generates Z3 constraint objects based on the provided context.

    Args:
        ctx: The pipeline context containing configuration and variables.

    Returns:
        A list of Z3 constraints (boolean expressions).
    """
    pass
