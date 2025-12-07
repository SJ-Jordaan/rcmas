from abc import ABC, abstractmethod
from src.core.state import PipelineContext

class Handler(ABC):
    @abstractmethod
    def handle(self, context: PipelineContext) -> PipelineContext:
        pass
