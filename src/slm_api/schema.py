from typing import List, Optional
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    messages: List[Message] = None
    reward: Optional[float] = None
    target_response: Optional[str] = None