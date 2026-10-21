from pydantic import BaseModel,Field

class LLMCallRecord(BaseModel):
    stage: str
    model: str
    latency: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None


class AgentResult(BaseModel):
    answer: str
    latency: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None
    call_records : list[LLMCallRecord] = Field(default_factory=list)