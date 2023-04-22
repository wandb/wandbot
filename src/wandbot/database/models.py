from sqlalchemy import Column, DateTime, Float, String

from .database import Base


class QuestionAnswers(Base):
    __tablename__ = "question_answers"

    question_answer_id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, primary_key=True, index=True)
    question = Column(String)
    answer = Column(String)
    feedback = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    time_taken = Column(Float)
