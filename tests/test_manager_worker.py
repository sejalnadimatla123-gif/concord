
import json

import pytest

from core.agents.result import AgentResult
from core.models import TaskPlan
from core.topologies.manager_worker import ManagerWorkerTopology
from core.agents.result import AgentResult, LLMCallRecord


class MockAgent:
    def __init__(self, answers):
        self.answers = iter(answers)

    async def run(self, task: str) -> AgentResult:
        answer = next(self.answers)

        return AgentResult(
            answer=answer,
            latency=0.1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.001,
            call_records=[
                LLMCallRecord(
                    stage="agent",
                    model="test-model",
                    latency=0.1,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cost=0.001,
                )
            ],
        )


@pytest.mark.asyncio
async def test_manager_worker_aggregates_metrics():
    manager = MockAgent([
        json.dumps({
            "subtasks": ["Find the causes", "Explain the effects"]
        }),
        "The causes and effects are summarized here.",
    ])

    worker = MockAgent([
        "Cause A",
        "Effect B",
    ])

    topology = ManagerWorkerTopology(manager, worker)
    result = await topology.run("Explain the causes and effects.")

    assert result.answer == "The causes and effects are summarized here."
    assert result.prompt_tokens == 40
    assert result.completion_tokens == 20
    assert result.total_tokens == 60
    assert result.cost == pytest.approx(0.004)
    assert result.latency >= 0
    assert len(result.call_records) == 4
    assert [record.stage for record in result.call_records] == [
        "planning",
        "worker",
        "worker",
        "synthesis",
    ]



@pytest.mark.asyncio
async def test_manager_worker_rejects_invalid_plan():
    manager = MockAgent(["This is not valid JSON"])
    worker = MockAgent([])

    topology = ManagerWorkerTopology(manager, worker)

    with pytest.raises(ValueError, match="invalid task plan"):
        await topology.run("Explain the causes and effects.")


@pytest.mark.asyncio
async def test_manager_worker_propagates_worker_failure():
    class FailingWorker:
        async def run(self, task: str) -> AgentResult:
            raise RuntimeError("Worker failed")

    manager = MockAgent([
        json.dumps({
            "subtasks": ["Find causes", "Explain effects"]
        })
    ])

    topology = ManagerWorkerTopology(manager, FailingWorker())

    with pytest.raises(RuntimeError, match="Worker failed"):
        await topology.run("Explain causes and effects.")
