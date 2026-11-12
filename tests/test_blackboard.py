import pytest

from core.agents.base import BaseAgent
from core.agents.result import AgentResult, LLMCallRecord
from core.topologies.blackboard import BlackboardTopology


class MockAgent(BaseAgent):
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    async def run(self, task: str) -> AgentResult:
        self.calls.append(task)

        return AgentResult(
            answer=self.answer,
            latency=0.01,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.001,
            call_records=[
                LLMCallRecord(
                    stage="agent",
                    model="test-model",
                    latency=0.01,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cost=0.001,
                )
            ],
        )



@pytest.mark.asyncio
async def test_blackboard_shares_knowledge_and_aggregates_metrics():
    agent_a = MockAgent("The answer is 425.")
    agent_b = MockAgent("Agent A's calculation is correct.")
    synthesizer = MockAgent("The final answer is 425.")

    topology = BlackboardTopology(
        agents=[agent_a, agent_b],
        synthesizer=synthesizer,
    )

    result = await topology.run("What is 25 * 17?")

    # Verify all agents were called.
    assert len(agent_a.calls) == 1
    assert len(agent_b.calls) == 1
    assert len(synthesizer.calls) == 1

    # Agent B should receive Agent A's contribution.
    assert "The answer is 425." in agent_b.calls[0]

    # Synthesizer should receive both contributions.
    assert "The answer is 425." in synthesizer.calls[0]
    assert "Agent A's calculation is correct." in synthesizer.calls[0]

    # Verify final answer.
    assert result.answer == "The final answer is 425."

    # Verify aggregated metrics (3 LLM calls).
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 15
    assert result.total_tokens == 45
    assert result.cost == pytest.approx(0.003)

    # Verify call-level tracing.
    assert len(result.call_records) == 3

    assert [record.stage for record in result.call_records] == [
        "contributor_1",
        "contributor_2",
        "synthesis",
    ]