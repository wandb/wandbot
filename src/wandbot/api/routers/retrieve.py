from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel
from starlette import status
from wandbot.retriever.base import Retriever

router = APIRouter(
    prefix="/retrieve",
    tags=["retrievers"],
)

retriever: Retriever | None = None


class APIRetrievalResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]


class APIRetrievalResponse(BaseModel):
    query: str
    top_k: List[APIRetrievalResult]


class APIRetrievalRequest(BaseModel):
    query: str
    language: str = "en"
    initial_k: int = 10
    top_k: int = 5
    include_tags: List[str] = []
    exclude_tags: List[str] = []


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
        indices=(
            [idx.value for idx in request.indices] if request.indices else None
        ),
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
