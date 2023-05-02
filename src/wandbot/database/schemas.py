from datetime import datetime
from enum import IntEnum

from pydantic import BaseModel


class Rating(IntEnum):
    positive = 1
    negative = -1
    neutral = 0


class FeedbackBase(BaseModel):
    rating: Rating | None = None


class Feedback(FeedbackBase):
    class Config:
        use_enum_values = True
        orm_mode = True


class FeedbackCreate(Feedback):
    feedback_id: str
    question_answer_id: str


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
    feedback: list[Feedback] | None = []


class QuestionAnswer(QuestionAnswerBase):
    class Config:
        use_enum_values = True
        orm_mode = True


class QuestionAnswerCreate(QuestionAnswer):
    question_answer_id: str
    thread_id: str


class ChatThreadBase(BaseModel):
    question_answers: list[QuestionAnswer] | None = []


class ChatThread(ChatThreadBase):
    application: str

    class Config:
        use_enum_values = True
        orm_mode = True


class ChatThreadCreate(ChatThread):
    thread_id: str
