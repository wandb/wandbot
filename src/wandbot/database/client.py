from sqlalchemy.future import create_engine
from sqlalchemy.orm import sessionmaker
from wandbot.database.config import DataBaseConfig
from wandbot.database.models import ChatThread as ChatThreadModel
from wandbot.database.models import QuestionAnswer as QuestionAnswerModel
from wandbot.database.schemas import ChatThread as ChatThreadSchema
from wandbot.database.schemas import Feedback as FeedbackSchema


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

    def get_chat_thread(self, thread_id: str) -> ChatThreadModel | None:
        chat_thread = (
            self.database.query(ChatThreadModel)
            .filter(ChatThreadModel.thread_id == thread_id)
            .first()
        )
        return chat_thread

    def create_chat_thread(self, chat_thread: ChatThreadSchema) -> ChatThreadModel:
        chat_thread = ChatThreadModel(
            thread_id=chat_thread.thread_id,
            application=chat_thread.application,
            question_answers=[
                QuestionAnswerModel(**question_answer.dict())
                for question_answer in chat_thread.question_answers
            ],
        )
        self.database.add(chat_thread)
        self.database.commit()
        self.database.refresh(chat_thread)
        return chat_thread

    def update_chat_thread(
        self, chat_thread: ChatThreadSchema
    ) -> ChatThreadModel | None:
        db_chat_thread = self.get_chat_thread(thread_id=chat_thread.thread_id)
        try:
            if chat_thread.question_answers and db_chat_thread:
                question_answers = [
                    QuestionAnswerModel(**schema_question_answer.dict())
                    for schema_question_answer in chat_thread.question_answers
                ]
                db_chat_thread.question_answers.extend(question_answers)
                self.database.flush()
                self.database.commit()
                self.database.refresh(db_chat_thread)
                return db_chat_thread
            else:
                return self.create_chat_thread(chat_thread=chat_thread)
        except Exception as e:
            self.database.rollback()
            return None

    def update_feedback(self, feedback: FeedbackSchema) -> QuestionAnswerModel:
        db_question_answer = (
            self.database.query(QuestionAnswerModel)
            .filter(
                QuestionAnswerModel.question_answer_id == feedback.question_answer_id,
                QuestionAnswerModel.thread_id == feedback.thread_id,
            )
            .first()
        )
        db_question_answer.feedback = feedback.feedback
        self.database.commit()
        self.database.refresh(db_question_answer)

        return db_question_answer
