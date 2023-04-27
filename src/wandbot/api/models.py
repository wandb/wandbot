from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


class ChatThread(Base):
    __tablename__ = "chat_thread"

    thread_id = Column(String, primary_key=True, index=True)
    application = Column(String)
    question_answers = relationship("QuestionAnswers", back_populates="chat_thread")


class QuestionAnswers(Base):
    __tablename__ = "question_answers"

    question_answer_id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, ForeignKey("chat_thread.thread_id"))
    question = Column(String)
    answer = Column(String)
    sources = Column(String)
    source_documents = Column(String)
    feedback = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    time_taken = Column(Float)
    total_tokens = Column(Integer)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    successful_requests = Column(Integer)
    total_cost = Column(Float)

    chat_thread = relationship("ChatThread", back_populates="question_answers")
