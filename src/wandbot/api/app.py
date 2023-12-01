"""This module serves as the main server API for the wandbot application.

It imports and uses the FastAPI framework to define the API and initialize application event handlers like "startup".
Also, the module includes Python's built-in asyncio library for managing asynchronous tasks related to database backup.

The API includes:
- APICreateChatThreadRequest
- APIFeedbackRequest
- APIFeedbackResponse
- APIGetChatThreadResponse
- APIQueryRequest
- APIQueryResponse
- APIQuestionAnswerRequest
- APIQuestionAnswerResponse

Following classes and their functionalities:
- Chat: Main chat handling class, initialized during startup.
- ChatConfig: Configuration utility for chat.
- ChatRequest: Schema to handle requests made to the chat.

It also sets up and interacts with the database through:
- DatabaseClient: A utility to interact with the database.
- Base.metadata.create_all(bind=engine): Creates database tables based on the metadata.

The server runs periodic backup of the data to wandb using the backup_db method which runs as a coroutine.
The backup data is transformed into a Pandas DataFrame and saved as a wandb.Table.

It uses logger from the utils module for logging purposes.
"""

import asyncio
from datetime import datetime, timezone

import pandas as pd
import wandb
from fastapi import FastAPI, Response, status

from wandbot.api.schemas import (
    APICreateChatThreadRequest,
    APIFeedbackRequest,
    APIFeedbackResponse,
    APIGetChatThreadResponse,
    APIQueryRequest,
    APIQueryResponse,
    APIQuestionAnswerRequest,
    APIQuestionAnswerResponse,
)
from wandbot.chat.chat import Chat
from wandbot.chat.config import ChatConfig
from wandbot.chat.schemas import ChatRequest
from wandbot.database.client import DatabaseClient
from wandbot.database.database import engine
from wandbot.database.models import Base
from wandbot.utils import get_logger

logger = get_logger(__name__)

Base.metadata.create_all(bind=engine)
chat: Chat | None = None
app = FastAPI(name="wandbot", version="1.0.0")
db_client: DatabaseClient | None = None
last_backup = datetime.now().astimezone(timezone.utc)


async def backup_db():
    """Periodically backs up the database to a table.

    This function runs periodically and retrieves all question-answer threads from the database since the last backup.
    It then creates a pandas DataFrame from the retrieved threads and logs it to a table using Weights & Biases.
    The last backup timestamp is updated after each backup.

    Returns:
        None
    """
    global last_backup
    while True:
        chat_threads = db_client.get_all_question_answers(last_backup)
        if chat_threads is not None:
            chat_table = pd.DataFrame(
                [chat_thread for chat_thread in chat_threads]
            )
            last_backup = datetime.now().astimezone(timezone.utc)
            logger.info(f"Backing up database to Table at {last_backup}")
            wandb.log(
                {"question_answers_db": wandb.Table(dataframe=chat_table)}
            )
        await asyncio.sleep(600)


@app.on_event("startup")
def startup_event():
    """Handles the startup event.

    This function initializes the chat and database client objects and creates a task to backup the database.

    Returns:
        None
    """
    global chat, db_client
    chat = Chat(ChatConfig())
    db_client = DatabaseClient()
    asyncio.create_task(backup_db())


@app.post(
    "/question_answer",
    response_model=APIQuestionAnswerResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_question_answer(
    request: APIQuestionAnswerRequest, response: Response
) -> APIQuestionAnswerResponse | None:
    """Creates a question answer.

    Args:
        request: The request object containing the question answer data.
        response: The response object to update with the result.

    Returns:
        The created question answer or None if creation failed.
    """
    question_answer = db_client.create_question_answer(request)
    if question_answer is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
    return question_answer


@app.get(
    "/chat_thread/{application}/{thread_id}",
    response_model=APIGetChatThreadResponse | None,
    status_code=status.HTTP_200_OK,
)
async def get_chat_thread(
    application: str, thread_id: str, response: Response
) -> APIGetChatThreadResponse:
    """Retrieves a chat thread from the database.

    If the chat thread does not exist, it creates a new chat thread.

    Args:
        application: The application name.
        thread_id: The ID of the chat thread.
        response: The HTTP response object.

    Returns:
        The retrieved or created chat thread.
    """
    chat_thread = db_client.get_chat_thread(
        application=application,
        thread_id=thread_id,
    )
    if chat_thread is None:
        chat_thread = db_client.create_chat_thread(
            APICreateChatThreadRequest(
                application=application,
                thread_id=thread_id,
            )
        )
        response.status_code = status.HTTP_201_CREATED
    if chat_thread is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
    return chat_thread


@app.post(
    "/query", response_model=APIQueryResponse, status_code=status.HTTP_200_OK
)
async def query(
    request: APIQueryRequest,
) -> APIQueryResponse:
    """Executes a query using the chat function and returns the result as an APIQueryResponse.

    Args:
        request: The APIQueryRequest object containing the question and chat history.

    Returns:
        The APIQueryResponse object containing the result of the query.
    """
    result = chat(
        ChatRequest(
            question=request.question,
            chat_history=request.chat_history,
            language=request.language,
            application=request.application,
        ),
    )
    result = APIQueryResponse(**result.model_dump())

    return result


@app.post(
    "/feedback",
    response_model=APIFeedbackResponse | None,
    status_code=status.HTTP_201_CREATED,
)
async def feedback(
    request: APIFeedbackRequest, response: Response
) -> APIFeedbackResponse:
    """Handles the feedback request and logs the feedback data.

    Args:
        request: The feedback request object.
        response: The response object.

    Returns:
        The feedback response object.
    """
    feedback_response = db_client.create_feedback(request)
    if feedback_response is not None:
        wandb.log(
            {
                "feedback": wandb.Table(
                    columns=list(request.model_dump().keys()),
                    data=[list(request.model_dump().values())],
                )
            }
        )
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
    return feedback_response


@app.on_event("shutdown")
def shutdown_event():
    """Finish the current run if wandb.run is not None.

    Returns:
        None
    """
    if wandb.run is not None:
        wandb.run.finish()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
