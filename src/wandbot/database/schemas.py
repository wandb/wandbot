"""This module defines the Pydantic models for the chat system.

This module contains the Pydantic models that are used to validate the data 
for the chat system. It includes models for chat threads, chat requests, and 
chat responses. The models are used to ensure that the data sent to and received 
from the chat system is in the correct format.

Typical usage example:

  chat_thread = ChatThread(thread_id="123", application="app1")
  chat_request = ChatRequest(question="What is the weather?", chat_history=None)
  chat_response = ChatResponse(system_prompt="Weather is sunny", question="What is the weather?",
                               answer="It's sunny", model="model1", sources="source1", 
                               source_documents="doc1", total_tokens=10, prompt_tokens=2, 
                               completion_tokens=8, time_taken=1.0, 
                               start_time=datetime.now(), end_time=datetime.now())
"""

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
        from_attributes = True


class FeedbackCreate(Feedback):
    feedback_id: str
    question_answer_id: str


class QuestionAnswerBase(BaseModel):
    system_prompt: str | None = None
    question: str
    answer: str | None = None
    model: str | None = None
    sources: str | None = None
    source_documents: str | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    time_taken: float | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    feedback: list[Feedback] | None = []
    language: str | None = None


class QuestionAnswer(QuestionAnswerBase):
    class Config:
        use_enum_values = True
        from_attributes = True


class QuestionAnswerCreate(QuestionAnswer):
    question_answer_id: str
    thread_id: str


class ChatThreadBase(BaseModel):
    question_answers: list[QuestionAnswer] | None = []


class ChatThread(ChatThreadBase):
    application: str

    class Config:
        use_enum_values = True
        from_attributes = True


class ChatThreadCreate(ChatThread):
    thread_id: str
