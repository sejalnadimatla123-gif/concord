
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.agents.solo_agent import SoloAgent


@pytest.mark.asyncio
async def test_solo_agent_records_llm_call():
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="425")
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        _hidden_params={"response_cost": 0.002},
    )

    with patch(
        "core.agents.solo_agent.completion",
        return_value=fake_response,
    ) as mock_completion:
        result = await SoloAgent().run("What is 25 * 17?")

    mock_completion.assert_called_once()

    assert result.answer == "425"
    assert result.total_tokens == 15
    assert result.cost == pytest.approx(0.002)

    assert len(result.call_records) == 1

    record = result.call_records[0]
    assert record.stage == "agent"
    assert record.model == "gemini/gemini-2.5-flash"
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 5
    assert record.total_tokens == 15
    assert record.cost == pytest.approx(0.002)
    assert record.latency >= 0
