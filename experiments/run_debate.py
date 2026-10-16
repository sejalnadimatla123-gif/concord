
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from core.agents.solo_agent import SoloAgent
from core.topologies.debate import DebateTopology
from experiments.record_results import save_result


# Load API credentials from the project-root .env file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


async def main():
    task = "What is 25 * 17? Explain the calculation briefly."

    # Three agent roles; currently all use the same Gemini model.
    agent_a = SoloAgent()
    agent_b = SoloAgent()
    judge = SoloAgent()

    topology = DebateTopology(
        agent_a=agent_a,
        agent_b=agent_b,
        judge=judge,
    )

    print(f"Task: {task}\n")
    print("Running DebateTopology (five LLM calls)...\n")

    result = await topology.run(task)

    print("FINAL ANSWER")
    print(result.answer)

    print("\nAGGREGATE METRICS")
    print(json.dumps(result.model_dump(), indent=2))

    save_result({
    "topology": "debate",
    "task": task,
    "llm_calls": 5,
    **result.model_dump(),
})


if __name__ == "__main__":
    asyncio.run(main())
