from fastapi import APIRouter
from starlette import status
from wandbot.chat.chat import Chat, ChatConfig
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.utils import get_logger

logger = get_logger(__name__)

chat_config = ChatConfig()
chat: Chat | None = None

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


class APIQueryRequest(ChatRequest):
    pass


class APIQueryResponse(ChatResponse):
    pass


@router.post(
    "/query", response_model=APIQueryResponse, status_code=status.HTTP_200_OK
)
def query(
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
