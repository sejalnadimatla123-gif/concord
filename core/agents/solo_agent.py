import time

from litellm import completion

from core.agents.base import BaseAgent
from core.agents.result import AgentResult, LLMCallRecord


class SoloAgent(BaseAgent):
    async def run(self, task: str) -> AgentResult:
        model = "gemini/gemini-2.5-flash"
        start_time = time.perf_counter()

        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": task}
            ],
        )

        latency = time.perf_counter() - start_time

        answer = response.choices[0].message.content
        usage = response.usage
        cost = response._hidden_params.get("response_cost")


        call_record = LLMCallRecord(
            stage="agent",
            model=model,
            latency=latency,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost,
        )

        return AgentResult(
            answer=answer,
            latency=latency,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost,
            call_records=[call_record],
        )