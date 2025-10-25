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
from typing import Dict, List

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
        from_attributes = True


class ChatRequest(BaseModel):
    question: str
    chat_history: List[QuestionAnswer] | None = None
    application: str | None = None
    language: str = "en"
    stream: bool = False


class ChatResponse(BaseModel):
    system_prompt: str
    question: str
    answer: str
    model: str
    sources: str
    source_documents: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    time_taken: float
    start_time: datetime
    end_time: datetime
    api_call_statuses: dict = {}
    response_synthesis_llm_messages: List[Dict[str, str]] | None = None
