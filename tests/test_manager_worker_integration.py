import pytest

from core.agents.solo_agent import SoloAgent
from core.topologies.manager_worker import ManagerWorkerTopology


@pytest.mark.asyncio
async def test_manager_worker_with_real_gemini():
    manager = SoloAgent()
    worker = SoloAgent()

    topology = ManagerWorkerTopology(
        manager=manager,
        worker=worker,
    )

    result = await topology.run(
        "Explain how the distributive property makes "
        "25 * 17 easier to calculate."
    )

    print(f"\nFinal answer: {result.answer}")
    print(f"Latency: {result.latency:.4f}s")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Cost: ${result.cost}")

    print("\nIndividual call records:")
    for record in result.call_records:
        print(record)

    assert result.answer
    assert result.total_tokens > 0
    assert result.latency > 0
    assert len(result.call_records) >= 4

    stages = [record.stage for record in result.call_records]

    assert stages[0] == "planning"
    assert stages[-1] == "synthesis"
    assert all(stage == "worker" for stage in stages[1:-1])