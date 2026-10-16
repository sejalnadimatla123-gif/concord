
import pytest

from core.agents.base import BaseAgent
from core.agents.result import AgentResult
from core.topologies.debate import DebateTopology


def make_result(answer: str, cost: float = 0.001) -> AgentResult:
    return AgentResult(
        answer=answer,
        latency=0.01,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost=cost,
    )


class MockAgent(BaseAgent):
    def __init__(self, answers: list[str]):
        self.answers = answers
        self.calls: list[str] = []

    async def run(self, task: str) -> AgentResult:
        self.calls.append(task)
        answer = self.answers.pop(0)
        return make_result(answer)


@pytest.mark.asyncio
async def test_debate_runs_five_calls_and_returns_judgment():
    agent_a = MockAgent(["Proposal A", "Review of B"])
    agent_b = MockAgent(["Proposal B", "Review of A"])
    judge = MockAgent(["Final answer"])

    topology = DebateTopology(agent_a, agent_b, judge)
    result = await topology.run("Solve this task")

    assert result.answer == "Final answer"
    assert len(agent_a.calls) == 2
    assert len(agent_b.calls) == 2
    assert len(judge.calls) == 1

    assert agent_a.calls[0] == "Solve this task"
    assert agent_b.calls[0] == "Solve this task"
    assert "Proposal B" in agent_a.calls[1]
    assert "Proposal A" in agent_b.calls[1]
    assert "Review of B" in judge.calls[0]
    assert "Review of A" in judge.calls[0]


@pytest.mark.asyncio
async def test_debate_aggregates_metrics():
    agent_a = MockAgent(["Proposal A", "Review of B"])
    agent_b = MockAgent(["Proposal B", "Review of A"])
    judge = MockAgent(["Final answer"])

    topology = DebateTopology(agent_a, agent_b, judge)
    result = await topology.run("Solve this task")

    assert result.prompt_tokens == 50
    assert result.completion_tokens == 25
    assert result.total_tokens == 75
    assert result.cost == pytest.approx(0.005)
    assert result.latency > 0


@pytest.mark.asyncio
async def test_debate_cost_is_unknown_if_any_call_cost_is_unknown():
    agent_a = MockAgent(["Proposal A", "Review of B"])
    agent_b = MockAgent(["Proposal B", "Review of A"])
    judge = MockAgent(["Final answer"])

    # Make one of the five calls return an unknown cost.
    original_run = agent_b.run

    async def run_with_unknown_cost(task: str) -> AgentResult:
        result = await original_run(task)
        return result.model_copy(update={"cost": None})

    agent_b.run = run_with_unknown_cost

    topology = DebateTopology(agent_a, agent_b, judge)
    result = await topology.run("Solve this task")

    assert result.answer == "Final answer"
    assert result.cost is None
