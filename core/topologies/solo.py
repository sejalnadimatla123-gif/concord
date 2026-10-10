from core.agents.base import BaseAgent
from core.agents.result import AgentResult


class SoloTopology:
    """Executes a task using a single agent."""

    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def run(self, task: str) -> AgentResult:
        return await self.agent.run(task)