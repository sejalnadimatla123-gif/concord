import pytest

from core.agents.base import BaseAgent
from core.topologies.solo import SoloTopology


class MockAgent(BaseAgent):
    async def run(self, task: str) -> str:
        return f"Mock response to: {task}"


@pytest.mark.asyncio
async def test_solo_topology():
    agent = MockAgent()
    topology = SoloTopology(agent)

    result = await topology.run("What is 25 * 17?")

    assert result == "Mock response to: What is 25 * 17?"