from typing import Any, List

from sqlalchemy.future import create_engine
from sqlalchemy.orm import sessionmaker
from wandbot.database.config import DataBaseConfig
from wandbot.database.models import ChatThread as ChatThreadModel
from wandbot.database.models import FeedBack as FeedBackModel
from wandbot.database.models import QuestionAnswer as QuestionAnswerModel
from wandbot.database.schemas import ChatThreadCreate as ChatThreadCreateSchema
from wandbot.database.schemas import Feedback as FeedbackSchema
from wandbot.database.schemas import QuestionAnswerCreate as QuestionAnswerCreateSchema


class Database:
    db_config = DataBaseConfig()

    def __init__(self, database: str | None = None):
        if database is not None:
            engine = create_engine(
                url=database, connect_args=self.db_config.connect_args
            )
        else:
            engine = create_engine(
                url=self.db_config.SQLALCHEMY_DATABASE_URL,
                connect_args=self.db_config.connect_args,
            )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def __get__(self, instance, owner):
        if not hasattr(self, "db"):
            self.db = self.SessionLocal()
        return self.db

    def __set__(self, instance, value):
        self.db = value

    def __set_name__(self, owner, name):
        self.name = name


class DatabaseClient:
    database = Database()

    def __init__(self, database: str | None = None):
        if database is not None:
            self.database = Database(database=database)

    def get_chat_thread(
        self, application: str, thread_id: str
    ) -> ChatThreadModel | None:
        chat_thread = (
            self.database.query(ChatThreadModel)
            .filter(
                ChatThreadModel.thread_id == thread_id,
                ChatThreadModel.application == application,
            )
            .first()
        )
        return chat_thread

    def create_chat_thread(
        self, chat_thread: ChatThreadCreateSchema
    ) -> ChatThreadModel:
        try:
            chat_thread = ChatThreadModel(
                thread_id=chat_thread.thread_id, application=chat_thread.application
            )
            self.database.add(chat_thread)
            self.database.flush()
            self.database.commit()
            self.database.refresh(chat_thread)

        except Exception as e:
            self.database.rollback()

        return chat_thread

    def get_question_answer(
        self, question_answer_id: str, thread_id: str
    ) -> QuestionAnswerModel | None:
        question_answer = (
            self.database.query(QuestionAnswerModel)
            .filter(
                QuestionAnswerModel.thread_id == thread_id,
                QuestionAnswerModel.question_answer_id == question_answer_id,
            )
            .first()
        )
        return question_answer

    def create_question_answer(
        self, question_answer: QuestionAnswerCreateSchema
    ) -> QuestionAnswerModel:
        try:
            question_answer = QuestionAnswerModel(**question_answer.dict())
            self.database.add(question_answer)
            self.database.flush()
            self.database.commit()
            self.database.refresh(question_answer)
        except Exception as e:
            self.database.rollback()
        return question_answer

    def get_feedback(self, question_answer_id: str) -> FeedBackModel | None:
        feedback = (
            self.database.query(FeedBackModel)
            .filter(FeedBackModel.question_answer_id == question_answer_id)
            .first()
        )
        return feedback

    def create_feedback(self, feedback: FeedbackSchema) -> FeedBackModel:
        if feedback.rating:
            try:
                feedback = FeedBackModel(**feedback.dict())
                self.database.add(feedback)
                self.database.flush()
                self.database.commit()
                self.database.refresh(feedback)
            except Exception as e:
                self.database.rollback()

            return feedback

    def get_all_question_answers(self, time=None) -> List[dict[str, Any]] | None:
        question_answers = self.database.query(QuestionAnswerModel)
        if time is not None:
            question_answers = question_answers.filter(
                QuestionAnswerModel.end_time >= time
            )
        question_answers = question_answers.all()
        if question_answers is not None:
            question_answers = [
                QuestionAnswerCreateSchema.from_orm(question_answer).dict()
                for question_answer in question_answers
            ]
            return question_answers
