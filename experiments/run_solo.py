
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from core.agents.solo_agent import SoloAgent
from core.topologies.solo import SoloTopology
from experiments.record_results import save_result

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


async def main():
    task = "What is 25 * 17? Explain the calculation briefly."

    topology = SoloTopology(agent=SoloAgent())

    print(f"Task: {task}\n")
    print("Running SoloTopology (one LLM call)...\n")

    result = await topology.run(task)

    print("FINAL ANSWER")
    print(result.answer)

    print("\nAGGREGATE METRICS")
    print(json.dumps(result.model_dump(), indent=2))

    save_result({
        "topology": "solo",
        "task": task,
        "llm_calls": 1,
        **result.model_dump(),
    })

if __name__ == "__main__":
    asyncio.run(main())

