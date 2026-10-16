
import json
from pathlib import Path
from typing import Any

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def save_result(record: dict[str, Any]) -> None:
    """Append one experiment run to the persistent results file."""

    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        with RESULTS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = {
            "experiment": "concord_runs",
            "status": "preliminary",
            "runs": [],
        }

    # Preserve the results we recorded manually earlier.
    if "runs" not in data:
        data["runs"] = data.pop("results", [])

    data["runs"].append(record)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"Experiment result saved to {RESULTS_PATH}")
