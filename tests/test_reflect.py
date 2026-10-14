import pytest

from core.agents.base import BaseAgent
from core.agents.result import AgentResult
from core.topologies.reflect import ReflectTopology


class MockAgent(BaseAgent):
    def __init__(self):
        self.calls = []

    async def run(self, task: str) -> AgentResult:
        self.calls.append(task)

        if len(self.calls) == 1:
            return AgentResult(
                answer="The answer is 420.",
                latency=0.01,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cost=0.001,
            )

        return AgentResult(
            answer="The correct answer is 425.",
            latency=0.02,
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            cost=0.002,
        )


@pytest.mark.asyncio
async def test_reflect_topology():
    agent = MockAgent()
    topology = ReflectTopology(agent)

    result = await topology.run("What is 25 * 17?")

    assert result.answer == "The correct answer is 425."
    assert len(agent.calls) == 2
    assert agent.calls[0] == "What is 25 * 17?"
    assert "The answer is 420." in agent.calls[1]


@pytest.mark.asyncio
async def test_reflect_aggregates_metrics():
    agent = MockAgent()
    topology = ReflectTopology(agent)

    result = await topology.run("What is 25 * 17?")

    assert result.prompt_tokens == 30
    assert result.completion_tokens == 15
    assert result.total_tokens == 45
    assert result.cost == pytest.approx(0.003)
    assert result.latency > 0




@pytest.mark.asyncio
async def test_reflect_propagates_second_call_failure():
    class FailingMockAgent(BaseAgent):
        def __init__(self):
            self.calls = 0

        async def run(self, task: str) -> AgentResult:
            self.calls += 1

            if self.calls == 1:
                return AgentResult(
                    answer="Initial answer",
                    latency=0.01,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cost=0.001,
                )

            raise RuntimeError("Reflection call failed")

    agent = FailingMockAgent()
    topology = ReflectTopology(agent)

    with pytest.raises(RuntimeError, match="Reflection call failed"):
        await topology.run("Test task")

    assert agent.calls == 2


@pytest.mark.asyncio
async def test_reflect_returns_none_cost_if_either_call_cost_is_unknown():
    class UnknownCostMockAgent(BaseAgent):
        def __init__(self):
            self.calls = 0

        async def run(self, task: str) -> AgentResult:
            self.calls += 1

            return AgentResult(
                answer=f"Answer {self.calls}",
                latency=0.01,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cost=0.001 if self.calls == 1 else None,
            )

    agent = UnknownCostMockAgent()
    topology = ReflectTopology(agent)

    result = await topology.run("Test task")

    assert result.answer == "Answer 2"
    assert result.cost is None
    assert result.total_tokens == 30
    assert agent.calls == 2

