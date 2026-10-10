from pydantic import BaseModel


class AgentResult(BaseModel):
    answer: str
    latency: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None