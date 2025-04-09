# from typing import Any, Dict, List

# from fastapi import APIRouter
# from pydantic import BaseModel
# from starlette import status

# from wandbot.retriever.base import SimpleRetrievalEngine

# router = APIRouter(
#     prefix="/retrieve",
#     tags=["retrievers"],
# )

# retriever: SimpleRetrievalEngine | None = None


# class APIRetrievalResult(BaseModel):
#     text: str
#     score: float
#     metadata: Dict[str, Any]


# class APIRetrievalResponse(BaseModel):
#     query: str
#     top_k: List[APIRetrievalResult]


# class APIRetrievalRequest(BaseModel):
#     query: str
#     language: str = "en"
#     top_k: int = 5
#     sources: List[str] | None = None


# @router.post(
#     "",
#     response_model=APIRetrievalResponse,
#     status_code=status.HTTP_200_OK,
# )
# @router.post(
#     "/",
#     response_model=APIRetrievalResponse,
#     status_code=status.HTTP_200_OK,
# )
# def retrieve(request: APIRetrievalRequest) -> APIRetrievalResponse:
#     """Retrieves the top k results for a given query.

#     Args:
#         request: The APIRetrievalRequest object containing the query and other parameters.

#     Returns:
#         The APIRetrievalResponse object containing the query and top k results.
#     """
#     results = retriever(
#         question=request.query,
#         language=request.language,
#         top_k=request.top_k,
#         sources=request.sources,
#     )

#     return APIRetrievalResponse(
#         query=request.query,
#         top_k=[
#             APIRetrievalResult(
#                 text=result["text"],
#                 score=result["score"],
#                 metadata=result["metadata"],
#             )
#             for result in results
#         ],
#     )
