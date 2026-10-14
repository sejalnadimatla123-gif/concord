
import time

from core.agents.base import BaseAgent
from core.agents.result import AgentResult
from core.topologies.base import BaseTopology


class ReflectTopology(BaseTopology):
    """Executes a task, reviews the initial answer, and returns a revision."""

    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def run(self, task: str) -> AgentResult:
        start_time = time.perf_counter()

        # First call: generate an initial answer.
        initial_result = await self.agent.run(task)

        # Second call: review the initial answer.
        reflection_task = (
            "Review the following answer to the original task.\n"
            "Correct any mistakes and return the best final answer.\n\n"
            f"Original task: {task}\n"
            f"Initial answer: {initial_result.answer}"
        )

        reflected_result = await self.agent.run(reflection_task)

        # Measure total elapsed time across both calls.
        latency = time.perf_counter() - start_time

        return AgentResult(
            answer=reflected_result.answer,
            latency=latency,
            prompt_tokens=(
                initial_result.prompt_tokens
                + reflected_result.prompt_tokens
            ),
            completion_tokens=(
                initial_result.completion_tokens
                + reflected_result.completion_tokens
            ),
            total_tokens=(
                initial_result.total_tokens
                + reflected_result.total_tokens
            ),
            cost=(
                initial_result.cost + reflected_result.cost
                if initial_result.cost is not None
                and reflected_result.cost is not None
                else None
            ),
        )
