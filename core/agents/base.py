from abc import ABC, abstractmethod

from core.agents.result import AgentResult


class BaseAgent(ABC):
    """Base interface for all Concord agents."""

    @abstractmethod
    async def run(self, task: str) -> AgentResult:
        """Execute a task and return structured execution data."""
        raise NotImplementedError