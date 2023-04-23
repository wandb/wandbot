import datetime
from typing import Optional

from pydantic import BaseModel


class QuestionAnswerBase(BaseModel):
    question_answer_id: str
    thread_id: str
    question: str
    answer: str | None
    sources: str | None
    feedback: str | None
    start_time: datetime.datetime | None
    end_time: datetime.datetime | None
    time_taken: float | None


class QuestionAnswerCreate(QuestionAnswerBase):
    pass


class QuestionAnswer(QuestionAnswerBase):
    class Config:
        orm_mode = True


class ChatThreadBase(BaseModel):
    thread_id: str
    application: str


class ChatThreadCreate(ChatThreadBase):
    pass


class ChatThread(ChatThreadBase):
    question_answers: list[QuestionAnswer] = []

    class Config:
        orm_mode = True


class APIQueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None
    application: Optional[str] = None


class APIQueryResponse(BaseModel):
    answer: str
    thread_id: str
    question_answer_id: str
    sources: Optional[str] = None
