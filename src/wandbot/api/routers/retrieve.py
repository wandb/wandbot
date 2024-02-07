from enum import Enum
from typing import Any, List

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
    metadata: dict[str, Any]


class APIRetrievalResponse(BaseModel):
    query: str
    top_k: List[APIRetrievalResult]


class Indices(str, Enum):
    """The indices available for retrieval."""

    DOCODILE_EN = "docodile_en"
    DOCODILE_JA = "docodile_ja"
    WANDB_EXAMPLES_CODE = "wandb_examples_code"
    WANDB_EXAMPLES_COLAB = "wandb_examples_colab"
    WANDB_SDK_CODE = "wandb_sdk_code"
    WANDB_SDK_TESTS = "wandb_sdk_tests"
    WEAVE_SDK_CODE = "weave_sdk_code"
    WEAVE_EXAMPLES = "weave_examples"
    WANDB_EDU_CODE = "wandb_edu_code"
    WEAVE_JS = "weave_js"
    FC_REPORTS = "fc_reports"


class APIRetrievalRequest(BaseModel):
    query: str
    indices: List[Indices] = []
    language: str = "en"
    initial_k: int = 10
    top_k: int = 5
    include_tags: List[str] = []
    exclude_tags: List[str] = []
    include_web_results: bool = True


@router.post(
    "/",
    response_model=APIRetrievalResponse,
    status_code=status.HTTP_200_OK,
)
def retrieve(request: APIRetrievalRequest) -> APIRetrievalResponse:
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
        include_web_results=request.include_web_results,
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
