from pydantic import BaseModel

class AgentResponse(BaseModel):
    answer: str
    sources: list[str] = []
    handoff: bool = False

class SessionState(BaseModel):
    messages: list[dict] = []
    current_order_id: str | None = None
