from pydantic import BaseModel

class RunRequest(BaseModel):
    role: str
    user_input: str = ""
