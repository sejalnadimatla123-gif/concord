from fastapi import FastAPI
from core.models import RunRequest
from core.agents.solo_agent import SoloAgent  
from core.topologies.solo import SoloTopology  

app = FastAPI(title = "Concord")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/run")
async def run(request: RunRequest):
    agent = SoloAgent()
    topology = SoloTopology(agent)

    result = await topology.run(request.task)

    return result