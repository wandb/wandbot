from fastapi import APIRouter, HTTPException
from starlette import status
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.utils import get_logger

logger = get_logger(__name__)


class APIQueryRequest(ChatRequest):
    pass


class APIQueryResponse(ChatResponse):
    pass


# Store initialization components
chat_components = {
    "vector_store": None,
    "chat_config": None,
    "chat": None,  # We'll store the actual Chat instance here
}

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/query", response_model=APIQueryResponse, status_code=status.HTTP_200_OK
)
async def query(request: APIQueryRequest) -> APIQueryResponse:
    if not chat_components.get("chat"):
        raise HTTPException(
            status_code=503, detail="Chat service is not yet initialized"
        )

    try:
        chat_instance = chat_components["chat"]
        result = await chat_instance.__acall__(
            ChatRequest(
                question=request.question,
                chat_history=request.chat_history,
                language=request.language,
                application=request.application,
            ),
        )
        return APIQueryResponse(**result.model_dump())
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(
            status_code=500, detail="Error processing chat query"
        )
