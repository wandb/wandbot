import datetime

from pydantic import BaseModel


class QuestionAnswerBase(BaseModel):
    question_answer_id: str
    thread_id: str
    question: str
    answer: str | None
    feedback: str | None
    start_time: datetime.datetime | None
    end_time: datetime.datetime | None
    time_taken: float | None


class QuestionAnswerCreate(QuestionAnswerBase):
    pass


class QuestionAnswer(QuestionAnswerCreate):
    class Config:
        orm_mode = True
