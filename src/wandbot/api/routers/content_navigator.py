import  httpx

from fastapi import APIRouter
from pydantic import BaseModel
from starlette import status

CONTENT_NAVIGATOR_ENDPOINT = "https://wandb-content-navigator.replit.app/get_content"

class ContentNavigatorRequest(BaseModel):
    """A user query to be used by the content navigator app"""

    user_id: str = None
    query: str

class ContentNavigatorResponse(BaseModel):
    """Response from the content navigator app"""

    slack_response: str
    rejected_slack_response: str = ""
    response_items_count: int = 0


router = APIRouter(
    prefix="/generate_content_suggestions",
    tags=["content-navigator"],
)


@router.post("/", response_model=ContentNavigatorResponse, status_code=status.HTTP_200_OK)
async def generate_content_suggestions(request: ContentNavigatorRequest):
    async with httpx.AsyncClient(timeout=1200.0) as content_client:
        response = await content_client.post(
            CONTENT_NAVIGATOR_ENDPOINT,
            json={"query": request.query, "user_id": request.user_id},
        )
        response_data = response.json()
    
    slack_response = response_data.get("slack_response", "")
    rejected_slack_response = response_data.get("rejected_slack_response", "")
    response_items_count = response_data.get("response_items_count", 0)

    if slack_response == "":
        slack_response = "It looks like there is an issue with the Content Navigator app at \
the moment, please try again later."

    return ContentNavigatorResponse(
        slack_response=slack_response,
        rejected_slack_response=rejected_slack_response,
        response_items_count=response_items_count,
    )