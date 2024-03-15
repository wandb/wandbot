"""This module provides a Database and DatabaseClient class for managing database operations.

The Database class provides a connection to the database and manages the session. It also provides methods for
getting and setting the current session object and the name of the database.

The DatabaseClient class uses an instance of the Database class to perform operations such as getting and creating
chat threads, question answers, and feedback from the database.

Typical usage example:

  db_client = DatabaseClient()
  chat_thread = db_client.get_chat_thread(application='app1', thread_id='123')
  question_answer = db_client.create_question_answer(question_answer=QuestionAnswerCreateSchema())
"""

import json
from datetime import datetime, timedelta
from typing import Any, Collection, List

from sqlalchemy.future import create_engine
from sqlalchemy.orm import sessionmaker
from wandbot.database.config import DataBaseConfig
from wandbot.database.models import ChatThread as ChatThreadModel
from wandbot.database.models import FeedBack as FeedBackModel
from wandbot.database.models import QuestionAnswer as QuestionAnswerModel
from wandbot.database.models import YoutubeChatThread as YoutubeChatThreadModel
from wandbot.database.schemas import ChatThreadCreate as ChatThreadCreateSchema
from wandbot.database.schemas import Feedback as FeedbackSchema
from wandbot.database.schemas import (
    QuestionAnswerCreate as QuestionAnswerCreateSchema,
)
from wandbot.database.schemas import (
    YoutubeAssistantThreadCreate as YoutubeAssistantThreadCreateSchema,
)
from wandbot.utils import get_logger

logger = get_logger(__name__)


class Database:
    """A class representing a database connection.

    This class provides a connection to the database and manages the session.

    Attributes:
        db_config: An instance of the DataBaseConfig class.
        SessionLocal: A sessionmaker object for creating sessions.
        db: The current session object.
        name: The name of the database.
    """

    db_config: DataBaseConfig = DataBaseConfig()

    def __init__(self, database: str | None = None):
        """Initializes the Database instance.

        Args:
            database: The URL of the database. If None, the default URL is used.
        """
        if database is not None:
            engine: Any = create_engine(
                url=database, connect_args=self.db_config.connect_args
            )
        else:
            engine: Any = create_engine(
                url=self.db_config.SQLALCHEMY_DATABASE_URL,
                connect_args=self.db_config.connect_args,
            )
        self.SessionLocal: Any = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )

    def __get__(self, instance, owner) -> Any:
        """Gets the current session object.

        Args:
            instance: The instance of the owner class.
            owner: The owner class.

        Returns:
            The current session object.
        """
        if not hasattr(self, "db"):
            self.db: Any = self.SessionLocal()
        return self.db

    def __set__(self, instance, value) -> None:
        """Sets the current session object.

        Args:
            instance: The instance of the owner class.
            value: The new session object.
        """
        if value is not None:
            engine: Any = create_engine(
                url=value, connect_args=self.db_config.connect_args
            )
        else:
            engine: Any = create_engine(
                url=self.db_config.SQLALCHEMY_DATABASE_URL,
                connect_args=self.db_config.connect_args,
            )
        self.SessionLocal: Any = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )

    def __set_name__(self, owner, name) -> None:
        """Sets the name of the database.

        Args:
            owner: The owner class.
            name: The name of the database.
        """
        self.name: str = name


class DatabaseClient:
    database: Database = Database()

    def __init__(self, database: str | None = None):
        """Initializes the DatabaseClient instance.

        Args:
            database: The URL of the database. If None, the default URL is used.
        """
        if database is not None:
            self.database = database

    def get_chat_thread(
        self, application: str, thread_id: str
    ) -> ChatThreadModel | None:
        """Gets a chat thread from the database.

        Args:
            application: The application name.
            thread_id: The ID of the chat thread.

        Returns:
            The chat thread model if found, None otherwise.
        """
        chat_thread: ChatThreadModel | None = (
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
        """Creates a chat thread in the database.

        Args:
            chat_thread: The chat thread to create.

        Returns:
            The created chat thread model.
        """
        try:
            chat_thread: ChatThreadModel = ChatThreadModel(
                thread_id=chat_thread.thread_id,
                application=chat_thread.application,
            )
            self.database.add(chat_thread)
            self.database.flush()
            self.database.commit()
            self.database.refresh(chat_thread)

        except Exception as e:
            logger.error(f"Create chat thread failed with error: {e}")
            self.database.rollback()

        return chat_thread

    def get_question_answer(
        self, question_answer_id: str, thread_id: str
    ) -> QuestionAnswerModel | None:
        """Gets a question answer from the database.

        Args:
            question_answer_id: The ID of the question answer.
            thread_id: The ID of the chat thread.

        Returns:
            The question answer model if found, None otherwise.
        """
        question_answer: QuestionAnswerModel | None = (
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
        """Creates a question answer in the database.

        Args:
            question_answer: The question answer to create.

        Returns:
            The created question answer model.
        """
        try:
            question_answer: QuestionAnswerModel = QuestionAnswerModel(
                **question_answer.model_dump()
            )
            self.database.add(question_answer)
            self.database.flush()
            self.database.commit()
            self.database.refresh(question_answer)
        except Exception as e:
            logger.error(f"Create question answer failed with error: {e}")
            self.database.rollback()
        return question_answer

    def get_feedback(self, question_answer_id: str) -> FeedBackModel | None:
        """Gets feedback from the database.

        Args:
            question_answer_id: The ID of the question answer.

        Returns:
            The feedback model if found, None otherwise.
        """
        feedback: FeedBackModel | None = (
            self.database.query(FeedBackModel)
            .filter(FeedBackModel.question_answer_id == question_answer_id)
            .first()
        )
        return feedback

    def create_feedback(self, feedback: FeedbackSchema) -> FeedBackModel:
        """Creates feedback in the database.

        Args:
            feedback: The feedback to create.

        Returns:
            The created feedback model.
        """
        if feedback.rating:
            try:
                feedback: FeedBackModel = FeedBackModel(**feedback.model_dump())
                self.database.add(feedback)
                self.database.flush()
                self.database.commit()
                self.database.refresh(feedback)
            except Exception as e:
                logger.error(f"Create feedback failed with error: {e}")
                self.database.rollback()

            return feedback

    def get_all_question_answers(
        self, time: Any = None
    ) -> List[dict[str, Any]] | None:
        """Gets all question answers from the database.

        Args:
            time: The time to filter the question answers by.

        Returns:
            A list of question answer dictionaries if found, None otherwise.
        """
        question_answers = self.database.query(QuestionAnswerModel)
        if time is not None:
            question_answers = question_answers.filter(
                QuestionAnswerModel.end_time >= time
            )
        question_answers = question_answers.all()
        if question_answers is not None:
            question_answers = [
                json.loads(
                    QuestionAnswerCreateSchema.model_validate(
                        question_answer
                    ).model_dump_json()
                )
                for question_answer in question_answers
            ]
            return question_answers

    def get_youtube_chat_thread(
        self, thread_id: str
    ) -> YoutubeChatThreadModel | None:
        """Gets a youtube chat thread from the database.

        Args:
            assistant_thread_id: The ID of the youtube chat thread.

        Returns:
            The youtube chat thread model if found, None otherwise.
        """
        youtube_assistant_thread: YoutubeChatThreadModel | None = (
            self.database.query(YoutubeChatThreadModel)
            .filter(YoutubeChatThreadModel.thread_id == thread_id)
            .first()
        )
        return youtube_assistant_thread

    def create_youtube_chat_thread(
        self, youtube_assistant_thread: YoutubeAssistantThreadCreateSchema
    ) -> YoutubeChatThreadModel:
        """Creates a youtube chat thread in the database.

        Args:
            youtube_assistant_thread: The youtube chat thread to create.

        Returns:
            The created youtube chat thread.
        """
        try:
            youtube_assistant_thread: YoutubeChatThreadModel = (
                YoutubeChatThreadModel(**youtube_assistant_thread.model_dump())
            )
            youtube_assistant_thread.created_at = datetime.utcnow()
            youtube_assistant_thread.updated_at = datetime.utcnow()
            self.database.add(youtube_assistant_thread)
            self.database.flush()
            self.database.commit()
            self.database.refresh(youtube_assistant_thread)
        except Exception as e:
            logger.error(f"Create youtube chat thread failed with error: {e}")
            self.database.rollback()
        return youtube_assistant_thread

    def update_youtube_chat_thread_time(
        self, thread_id: str
    ) -> YoutubeChatThreadModel:
        """Updates a youtube chat thread in the database.

        Args:
            youtube_assistant_thread: The youtube chat thread to update.

        Returns:
            The updated youtube chat thread.
        """
        youtube_assistant_thread: YoutubeChatThreadModel = (
            self.get_youtube_chat_thread(thread_id=thread_id)
        )
        try:
            youtube_assistant_thread.updated_at = datetime.utcnow()
            self.database.add(youtube_assistant_thread)
            self.database.flush()
            self.database.commit()
            self.database.refresh(youtube_assistant_thread)
        except Exception as e:
            logger.error(f"Update youtube chat thread failed with error: {e}")
            self.database.rollback()
        return youtube_assistant_thread

    def get_old_youtube_chat_threads(
        self, time_delta: timedelta = timedelta(hours=24)
    ) -> Collection[YoutubeAssistantThreadCreateSchema]:
        """Deletes YoutubeAssistantThread records that have not been updated in the last 24 hours."""
        cutoff_time = datetime.now() - time_delta
        old_threads = self.database.query(YoutubeChatThreadModel).filter(
            YoutubeChatThreadModel.updated_at < cutoff_time
        )
        threads = [
            YoutubeAssistantThreadCreateSchema.model_validate(thread)
            for thread in old_threads.all()
        ]

        return threads

    def delete_youtube_chat_thread(self, thread_id: str) -> None:
        """Deletes a youtube chat thread from the database.

        Args:
            thread_id: The ID of the youtube chat thread to delete.
        """
        youtube_assistant_thread: YoutubeChatThreadModel = (
            self.get_youtube_chat_thread(thread_id=thread_id)
        )
        try:
            self.database.delete(youtube_assistant_thread)
            self.database.commit()
        except Exception as e:
            logger.error(f"Delete youtube chat thread failed with error: {e}")
            self.database.rollback()
