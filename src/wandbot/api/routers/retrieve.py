from fastapi import APIRouter
from starlette import status
from wandbot.api.schemas import (
    APIRetrievalRequest,
    APIRetrievalResponse,
    APIRetrievalResult,
)
from wandbot.retriever.base import Retriever

router = APIRouter(
    prefix="/retrieve",
    tags=["retrievers"],
)

retriever: Retriever | None = None


@router.post(
    "/",
    response_model=APIRetrievalResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve(request: APIRetrievalRequest) -> APIRetrievalResponse:
    """Retrieves the top k results for a given query.

    Args:
        request: The APIRetrievalRequest object containing the query and other parameters.

    Returns:
        The APIRetrievalResponse object containing the query and top k results.
    """
    results = retriever(
        query=request.query,
        language=request.language,
        top_k=request.top_k,
        include_tags=request.include_tags,
        exclude_tags=request.exclude_tags,
    )

    return APIRetrievalResponse(
        query=request.query,
        top_k=[
            APIRetrievalResult(
                text=result["text"],
                score=result["score"],
                metadata=result["metadata"],
            )
            for result in results
        ],
    )
