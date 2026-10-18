
import asyncio
import json
from pathlib import Path
from experiments.record_results import save_result

from dotenv import load_dotenv

from core.agents.solo_agent import SoloAgent
from core.topologies.manager_worker import ManagerWorkerTopology

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


async def main():
    task = (
        "Explain two major causes of ocean pollution "
        "and one consequence of each."
    )

    topology = ManagerWorkerTopology(
        manager=SoloAgent(),
        worker=SoloAgent(),
    )

    print(f"Task: {task}\n")
    print("Running ManagerWorkerTopology with Gemini...\n")

    result = await topology.run(task)

    print("FINAL ANSWER")
    print(result.answer)

    print("\nAGGREGATE METRICS")
    print(json.dumps(result.model_dump(), indent=2))

    save_result({
    "topology": "manager_worker",
    "task": task,
    "llm_calls": None,  # We'll log the exact count in a later improvement.
    **result.model_dump(),
})


if __name__ == "__main__":
    asyncio.run(main())
