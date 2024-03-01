import wandb
from fastapi import APIRouter
from starlette import status
from starlette.responses import Response

from wandbot.database.client import DatabaseClient
from wandbot.database.database import engine
from wandbot.database.models import Base
from wandbot.database.schemas import (
    ChatThread,
    ChatThreadCreate,
    Feedback,
    FeedbackCreate,
    QuestionAnswer,
    QuestionAnswerCreate,
    YoutubeAssistantThreadCreate,
    YoutubeAssistantThread,
)
from wandbot.utils import get_logger

logger = get_logger(__name__)

Base.metadata.create_all(bind=engine)

db_client: DatabaseClient | None = None

router = APIRouter(
    prefix="/data",
    tags=["database", "crud"],
)


class APIQuestionAnswerRequest(QuestionAnswerCreate):
    pass


class APIQuestionAnswerResponse(QuestionAnswer):
    pass


@router.post(
    "/question_answer",
    response_model=APIQuestionAnswerResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_question_answer(
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


class APIGetChatThreadResponse(ChatThread):
    pass


class APIGetChatThreadRequest(ChatThreadCreate):
    pass


class APICreateChatThreadRequest(ChatThreadCreate):
    pass


@router.get(
    "/chat_thread/{application}/{thread_id}",
    response_model=APIGetChatThreadResponse | None,
    status_code=status.HTTP_200_OK,
)
def get_chat_thread(
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


class APIFeedbackRequest(FeedbackCreate):
    pass


class APIFeedbackResponse(Feedback):
    pass


@router.post(
    "/feedback",
    response_model=APIFeedbackResponse | None,
    status_code=status.HTTP_201_CREATED,
)
def feedback(
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


class APIYouTubeAssistantThreadRequest(YoutubeAssistantThreadCreate):
    pass


class APIYouTubeAssistantThreadResponse(YoutubeAssistantThread):
    pass


@router.post(
    "/youtube_assistant_thread",
    response_model=APIYouTubeAssistantThreadResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_youtube_assistant_thread(
    request: APIYouTubeAssistantThreadRequest, response: Response
) -> APIYouTubeAssistantThreadResponse:
    """Creates a YouTube assistant thread.

    Args:
        request: The request object containing the YouTube assistant thread data.
        response: The response object to update with the result.

    Returns:
        The created YouTube assistant thread.

    """
    youtube_assistant_thread = db_client.create_youtube_assistant_thread(
        request
    )
    if youtube_assistant_thread is None:
        response.status_code = status.HTTP_400_BAD_REQUEST

    return youtube_assistant_thread


@router.get(
    "/youtube_assistant_thread/{thread_id}",
    response_model=APIYouTubeAssistantThreadResponse | None,
    status_code=status.HTTP_200_OK,
)
def get_youtube_assistant_thread(
    thread_id: str, response: Response
) -> APIYouTubeAssistantThreadResponse:
    """Retrieves a YouTube assistant thread from the database.

    Args:
        thread_id: The ID of the YouTube assistant thread.
        response: The HTTP response object.

    Returns:
        The retrieved YouTube assistant thread.
    """
    youtube_assistant_thread = db_client.get_youtube_assistant_thread(
        thread_id=thread_id
    )
    if youtube_assistant_thread is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
    return youtube_assistant_thread
