import json
import time

from core.agents.base import BaseAgent
from core.agents.result import AgentResult
from core.topologies.base import BaseTopology


class Blackboard:
    """Shared knowledge space for agents."""

    def __init__(self):
        self.entries: list[dict[str, str]] = []

    def write(self, agent_name: str, contribution: str):
        self.entries.append({
            "agent": agent_name,
            "contribution": contribution,
        })

    def read(self) -> list[dict[str, str]]:
        return self.entries


class BlackboardTopology(BaseTopology):
    """Coordinates agents through a shared knowledge space."""

    def __init__(
        self,
        agents: list[BaseAgent],
        synthesizer: BaseAgent,
    ):
        self.agents = agents
        self.synthesizer = synthesizer

    async def run(self, task: str) -> AgentResult:
        start_time = time.perf_counter()

        board = Blackboard()
        results = []

        # Phase 1: Agents contribute to the shared blackboard.
        for index, agent in enumerate(self.agents, start=1):
            existing_knowledge = json.dumps(board.read(), indent=2)

            prompt = (
                "You are contributing to a shared knowledge space.\n"
                "Solve the original task using the existing contributions "
                "where useful. You may extend, correct, or challenge them.\n\n"
                f"Original task: {task}\n\n"
                f"Existing blackboard contributions:\n"
                f"{existing_knowledge}\n\n"
                "Provide your contribution."
            )

            result = await agent.run(prompt)
            results.append((result, f"contributor_{index}"))

            board.write(f"Agent {index}", result.answer)

        # Phase 2: Synthesize all accumulated knowledge.
        synthesis_prompt = (
            "Produce the best final answer using the shared knowledge below.\n"
            "Resolve contradictions and correct any mistakes.\n\n"
            f"Original task: {task}\n\n"
            f"Blackboard contributions:\n"
            f"{json.dumps(board.read(), indent=2)}"
        )

        final_result = await self.synthesizer.run(synthesis_prompt)
        results.append((final_result, "synthesis"))

        # Phase 3: Aggregate metrics and call records.
        all_results = [result for result, _ in results]

        total_cost = (
            sum(result.cost for result in all_results)
            if all(result.cost is not None for result in all_results)
            else None
        )

        call_records = []

        for result, stage in results:
            for record in result.call_records:
                call_records.append(
                    record.model_copy(update={"stage": stage})
                )

        return AgentResult(
            answer=final_result.answer,
            latency=time.perf_counter() - start_time,
            prompt_tokens=sum(r.prompt_tokens for r in all_results),
            completion_tokens=sum(r.completion_tokens for r in all_results),
            total_tokens=sum(r.total_tokens for r in all_results),
            cost=total_cost,
            call_records=call_records,
        )
