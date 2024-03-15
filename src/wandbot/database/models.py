"""This module defines the SQLAlchemy models for the ChatThread, QuestionAnswer, and FeedBack tables.

Each class represents a table in the database and includes columns and relationships. The Base class is a declarative
base that stores a catalog of classes and mapped tables in the Declarative system.

Typical usage example:

  from wandbot.database.models import ChatThread, QuestionAnswer, FeedBack
  chat_thread = ChatThread(thread_id='123', application='app1')
  question_answer = QuestionAnswer(question_answer_id='456', thread_id='123')
  feedback = FeedBack(feedback_id='789', question_answer_id='456')
"""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ChatThread(Base):
    __tablename__ = "chat_thread"

    thread_id = Column(String, primary_key=True, index=True)
    application = Column(String)
    question_answers = relationship(
        "QuestionAnswer", back_populates="chat_thread"
    )


class QuestionAnswer(Base):
    __tablename__ = "question_answers"
    thread_id = Column(String, ForeignKey("chat_thread.thread_id"))
    question_answer_id = Column(String, primary_key=True, index=True)
    system_prompt = Column(String)
    question = Column(String)
    answer = Column(String)
    model = Column(String)
    sources = Column(String)
    source_documents = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    time_taken = Column(Float)
    total_tokens = Column(Integer)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    successful_requests = Column(Integer)
    total_cost = Column(Float)
    chat_thread = relationship("ChatThread", back_populates="question_answers")
    feedback = relationship("FeedBack", back_populates="question_answer")
    language = Column(String)


class FeedBack(Base):
    __tablename__ = "feedback"

    feedback_id = Column(String, primary_key=True, index=True)
    question_answer_id = Column(
        String, ForeignKey("question_answers.question_answer_id")
    )
    rating = Column(Integer)
    question_answer = relationship("QuestionAnswer", back_populates="feedback")


class YoutubeChatThread(Base):
    __tablename__ = "youtube_assistant"

    thread_id = Column(String, primary_key=True, index=True)
    video_id = Column(String)
    transcript = Column(String)
    summary = Column(String)
    key_points = Column(String)
