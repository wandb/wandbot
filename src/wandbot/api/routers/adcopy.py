from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel
from starlette import status

from wandbot.adcopy.adcopy import AdCopyEngine


class Action(Enum):
    Awareness = "awareness"
    SignUps = "signups"


class Persona(Enum):
    Technical = "technical"
    Executive = "executive"


class AdCopyRequest(BaseModel):
    action: Action
    persona: Persona
    query: str


class AdCopyResponse(BaseModel):
    ad_copies: str


router = APIRouter(
    prefix="/generate_ads",
    tags=["adcopy"],
)

ads_engine: AdCopyEngine | None = None


@router.post("/", response_model=AdCopyResponse, status_code=status.HTTP_200_OK)
def generate_ads(request: AdCopyRequest):
    response = ads_engine(
        query=request.query,
        persona=request.persona.value,
        action=request.action.value,
    )

    return AdCopyResponse(ad_copies=response)
