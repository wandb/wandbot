import datetime
from enum import Enum

from pydantic import BaseModel, BaseSettings, Field


class DataBaseConfig(BaseSettings):
    SQLALCHEMY_DATABASE_URL: str = Field(
        "sqlite:///./app.db", env="SQLALCHEMY_DATABASE_URL"
    )

    class Config:
        env_file = ".env"


class Feedback(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class QuestionAnswerBase(BaseModel):
    question_answer_id: str
    thread_id: str
    question: str
    answer: str | None
    sources: str | None
    source_documents: str | None = None
    feedback: Feedback | None
    start_time: datetime.datetime | None
    end_time: datetime.datetime | None
    time_taken: float | None

    class Config:
        use_enum_values = True


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
    question_answer_id: str | None = None
    thread_id: str | None = None
    application: str | None = None


class APIQueryResponse(BaseModel):
    answer: str
    thread_id: str
    question_answer_id: str
    sources: str | None = None
    source_documents: str | None = None


class FeedbackBase(BaseModel):
    feedback: Feedback
    question_answer_id: str
    thread_id: str

    class Config:
        use_enum_values = True


class APIFeedbackRequest(FeedbackBase):
    pass

    class Config:
        use_enum_values = True
