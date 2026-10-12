
from abc import ABC, abstractmethod

from core.agents.result import AgentResult


class BaseTopology(ABC):
    """Base interface for all Concord topologies."""

    @abstractmethod
    async def run(self, task: str) -> AgentResult:
        """Execute a task using a particular topology."""
        raise NotImplementedError
