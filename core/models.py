from pydantic import BaseModel

class RunRequest(BaseModel):
    task : str