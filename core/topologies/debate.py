import time

from core.agents.base import BaseAgent
from core.agents.result import AgentResult
from core.topologies.base import BaseTopology


class DebateTopology(BaseTopology):
    """Uses independent proposals, cross-review, and a judge to solve a task."""

    def __init__(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        judge: BaseAgent,
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.judge = judge

    async def run(self, task: str) -> AgentResult:
        start_time = time.perf_counter()

        # Phase 1: Both agents independently solve the original task.
        proposal_a = await self.agent_a.run(task)
        proposal_b = await self.agent_b.run(task)

        # Phase 2: Each agent reviews the other agent's proposal.
        review_a_task = (
            "Review the following proposed answer. Identify errors, "
            "unsupported claims, or missing reasoning. Be specific.\n\n"
            f"Original task: {task}\n"
            f"Your proposal: {proposal_a.answer}\n"
            f"Other agent's proposal: {proposal_b.answer}"
        )
        review_b_task = (
            "Review the following proposed answer. Identify errors, "
            "unsupported claims, or missing reasoning. Be specific.\n\n"
            f"Original task: {task}\n"
            f"Your proposal: {proposal_b.answer}\n"
            f"Other agent's proposal: {proposal_a.answer}"
        )

        review_a = await self.agent_a.run(review_a_task)
        review_b = await self.agent_b.run(review_b_task)

        # Phase 3: The judge considers the proposals and both reviews.
        judge_task = (
            "Produce the best final answer to the original task. "
            "Compare both proposals, consider the critiques, and correct "
            "any identified mistakes. Do not assume either agent is correct "
            "without evaluating its reasoning.\n\n"
            f"Original task: {task}\n\n"
            f"Proposal A: {proposal_a.answer}\n"
            f"Proposal B: {proposal_b.answer}\n\n"
            f"Review of A: {review_a.answer}\n"
            f"Review of B: {review_b.answer}"
        )

        final_result = await self.judge.run(judge_task)

        # Aggregate metrics across every call made by this topology.
        results = [
            proposal_a,
            proposal_b,
            review_a,
            review_b,
            final_result,
        ]

        total_cost = (
            sum(result.cost for result in results)
            if all(result.cost is not None for result in results)
            else None
        )

        return AgentResult(
            answer=final_result.answer,
            latency=time.perf_counter() - start_time,
            prompt_tokens=sum(r.prompt_tokens for r in results),
            completion_tokens=sum(r.completion_tokens for r in results),
            total_tokens=sum(r.total_tokens for r in results),
            cost=total_cost,
        )
