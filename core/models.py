from pydantic import BaseModel

class RunRequest(BaseModel):
    task : str

class TaskPlan(BaseModel):
    subtasks: list[str]