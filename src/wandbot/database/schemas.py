from datetime import datetime
from enum import Enum

from pydantic import BaseModel

#####################
# 1. Feedback
#####################


class Rating(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class FeedbackBase(BaseModel):
    feedback: Rating

    class Config:
        use_enum_values = True


class FeedbackCreate(FeedbackBase):
    question_answer_id: str
    thread_id: str


class Feedback(FeedbackCreate):
    class Config:
        use_enum_values = True


#####################
# Question Answer
#####################


class QuestionAnswerBase(BaseModel):
    question: str
    answer: str | None = None
    model: str | None = None
    sources: str | None = None
    source_documents: str | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    successful_requests: int | None = None
    total_cost: float | None = None
    time_taken: float | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    feedback: Rating | None = None

    class Config:
        use_enum_values = True


class QuestionAnswerCreate(QuestionAnswerBase):
    question_answer_id: str
    thread_id: str


class QuestionAnswer(QuestionAnswerCreate):
    class Config:
        orm_mode = True


#####################
# Chat Thread
#####################


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
