import pytest

from core.agents.solo_agent import SoloAgent


@pytest.mark.asyncio
async def test_solo_agent():
    agent = SoloAgent()

    result = await agent.run("What is 25 * 17?")

    print(f"\nAgent response: {result.answer}")
    print(f"Latency: {result.latency:.4f}s")
    print(f"Tokens: {result.total_tokens}")
    print(f"Cost: ${result.cost}")

    assert result.answer
    assert result.total_tokens > 0
    assert result.latency > 0
    assert result.cost is not None