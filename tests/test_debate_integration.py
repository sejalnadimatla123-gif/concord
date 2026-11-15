
import pytest

from core.agents.solo_agent import SoloAgent
from core.topologies.debate import DebateTopology


@pytest.mark.asyncio
async def test_debate_with_real_gemini():
    agent_a = SoloAgent()
    agent_b = SoloAgent()
    judge = SoloAgent()

    topology = DebateTopology(
        agent_a=agent_a,
        agent_b=agent_b,
        judge=judge,
    )

    result = await topology.run(
        "Explain why breaking 25 * 17 into "
        "(25 * 10) + (25 * 7) makes calculation easier."
    )

    print("\nFinal answer:", result.answer)
    print("Latency:", result.latency)
    print("Total tokens:", result.total_tokens)
    print("Cost:", result.cost)

    print("\nIndividual call records:")
    for record in result.call_records:
        print(record)

    assert result.answer
    assert result.total_tokens > 0
    assert result.latency > 0
    assert len(result.call_records) == 5

    stages = [record.stage for record in result.call_records]

    assert stages == [
        "proposal_a",
        "proposal_b",
        "review_a",
        "review_b",
        "judge",
    ]
