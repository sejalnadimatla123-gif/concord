import pytest

from core.agents.solo_agent import SoloAgent
from core.topologies.blackboard import BlackboardTopology


@pytest.mark.asyncio
async def test_blackboard_with_real_gemini():
    agents = [SoloAgent(), SoloAgent()]
    synthesizer = SoloAgent()

    topology = BlackboardTopology(
        agents=agents,
        synthesizer=synthesizer,
    )

    result = await topology.run(
        "Explain why breaking 25 * 17 into "
        "(25 * 10) + (25 * 7) makes calculation easier."
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
    assert len(result.call_records) == 3

    assert [r.stage for r in result.call_records] == [
        "contributor_1",
        "contributor_2",
        "synthesis",
    ]