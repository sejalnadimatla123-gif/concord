
import json
import time
from core.topologies.base import BaseTopology

from core.agents.base import BaseAgent
from core.models import TaskPlan
from core.agents.result import AgentResult


class TaskManager:
    """Breaks a complex task into structured subtasks."""

    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def decompose(self, task: str) -> tuple[TaskPlan , AgentResult]:
        prompt = (
            "Break the user's task into 2 to 4 independent subtasks. "
            "Return ONLY valid JSON in this exact format: "
            '{"subtasks": ["first subtask", "second subtask"]}\n\n'
            f"Task: {task}"
        )

        result = await self.agent.run(prompt)

        try:
            data = json.loads(result.answer)
            plan = TaskPlan.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "The manager returned an invalid task plan."
            ) from exc

        if not 2 <= len(plan.subtasks) <= 4:
            raise ValueError("The manager must produce 2 to 4 subtasks.")

        if any(not subtask.strip() for subtask in plan.subtasks):
            raise ValueError("Subtasks cannot be empty.")

        return plan,result


    async def synthesize(
            self,
            task: str,
            subtasks: list[str],
            results: list[AgentResult],
        ) -> AgentResult:
            prompt = (
                "Answer the original task using the worker results below. "
                "Resolve contradictions and provide one clear final answer.\n\n"
                f"Original task: {task}\n\n"
            )
    
            for subtask, result in zip(subtasks, results):
                prompt += (
                    f"Subtask: {subtask}\n"
                    f"Worker answer: {result.answer}\n\n"
                )
    
            return await self.agent.run(prompt)
    
    


class WorkerPool:
    """Executes planned subtasks using a worker agent."""

    def __init__(self, worker: BaseAgent):
        self.worker = worker

    async def execute(self, subtasks: list[str]) -> list[AgentResult]:
        results = []

        for subtask in subtasks:
            result = await self.worker.run(subtask)
            results.append(result)

        return results




class ManagerWorkerTopology(BaseTopology):
    """Decomposes tasks, executes subtasks, and synthesizes results."""

    def __init__(self, manager: BaseAgent, worker: BaseAgent):
        self.manager = TaskManager(manager)
        self.workers = WorkerPool(worker)

    async def run(self, task: str) -> AgentResult:
        start_time = time.perf_counter()

        # 1. Manager decomposes the task.
        plan, planning_result = await self.manager.decompose(task)

        # 2. Workers execute the subtasks.
        worker_results = await self.workers.execute(plan.subtasks)

        # 3. Manager synthesizes the final answer.
        final_result = await self.manager.synthesize(
            task, plan.subtasks, worker_results
        )

        # 4. Aggregate metrics from every LLM call.
        all_results = [planning_result, *worker_results, final_result]

        total_cost = (
            sum(result.cost for result in all_results)
            if all(result.cost is not None for result in all_results)
            else None
        )

        return AgentResult(
            answer=final_result.answer,
            latency=time.perf_counter() - start_time,
            prompt_tokens=sum(r.prompt_tokens for r in all_results),
            completion_tokens=sum(r.completion_tokens for r in all_results),
            total_tokens=sum(r.total_tokens for r in all_results),
            cost=total_cost,
        )


