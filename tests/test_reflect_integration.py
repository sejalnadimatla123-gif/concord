import pytest

from core.agents.solo_agent import SoloAgent
from core.topologies.reflect import ReflectTopology


@pytest.mark.asyncio
async def test_reflect_with_real_gemini():
    agent = SoloAgent()
    topology = ReflectTopology(agent)

    result = await topology.run("What is 25 * 17? Explain briefly.")

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
    assert len(result.call_records) == 2

    assert [r.stage for r in result.call_records] == [
        "initial_answer",
        "reflection",
    ]