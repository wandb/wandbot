from datetime import datetime
from typing import List

from pydantic import BaseModel
from wandbot.database.schemas import QuestionAnswer


class ChatThreadBase(BaseModel):
    question_answers: list[QuestionAnswer] | None = []


class ChatThreadCreate(ChatThreadBase):
    thread_id: str
    application: str

    class Config:
        use_enum_values = True


class ChatThread(ChatThreadCreate):
    class Config:
        orm_mode = True


class ChatRequest(BaseModel):
    question: str
    chat_history: List[QuestionAnswer] | None = None


class ChatRepsonse(BaseModel):
    question: str
    answer: str
    model: str
    sources: str
    source_documents: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    successful_requests: int
    total_cost: float
    time_taken: float
    start_time: datetime
    end_time: datetime
