import pytest

from core.agents.base import BaseAgent
from core.topologies.solo import SoloTopology
from core.topologies.base import BaseTopology
from core.agents.result import AgentResult


class MockAgent(BaseAgent):
    async def run(self, task: str) -> AgentResult:
        return AgentResult(
            answer=f"Mock response to: {task}",
            latency=0.01,
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            cost=0.0,
        )
    


@pytest.mark.asyncio
async def test_solo_topology():
    agent = MockAgent()
    topology = SoloTopology(agent)

    result = await topology.run("What is 25 * 17?")

    assert result.answer == "Mock response to: What is 25 * 17?"


def test_base_topology_is_abstract():
    with pytest.raises(TypeError):
        BaseTopology()  # Should raise TypeError because it's abstract