import datetime

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
