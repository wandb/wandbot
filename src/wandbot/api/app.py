import logging
from typing import Optional

from fastapi import FastAPI, Response, status
from wandbot.api.schemas import (
    APIFeedbackRequest,
    APIFeedbackResponse,
    APIQueryRequest,
    APIQueryResponse,
)
from wandbot.chat import Chat
from wandbot.config import ChatConfig, ChatRequest
from wandbot.database import models, schemas
from wandbot.database.client import DatabaseClient
from wandbot.database.database import engine
from wandbot.database.schemas import QuestionAnswerCreate

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)
chat: Optional[Chat] = None
app = FastAPI(name="wandbot", version="0.0.1")
db_client = DatabaseClient()


@app.on_event("startup")
def startup_event():
    global chat
    chat = Chat(ChatConfig())


@app.post("/chat_thread", status_code=status.HTTP_201_CREATED)
async def create_chat_thread(request: schemas.ChatThread, response: Response):
    chat_thread = db_client.update_chat_thread(request)
    if chat_thread is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
    return chat_thread


@app.get(
    "/chat_thread/{thread_id}",
    response_model=schemas.ChatThread | None,
    status_code=status.HTTP_200_OK,
)
async def get_chat_thread(thread_id: str, response: Response):
    chat_thread = db_client.get_chat_thread(
        thread_id,
    )
    if chat_thread is None:
        response.status_code = status.HTTP_404_NOT_FOUND
    return chat_thread


def get_chat_history(
    chat_history: list[QuestionAnswerCreate] | None,
) -> list[tuple[str, str]]:
    if not chat_history:
        return []
    else:
        return [
            (question_answer.question, question_answer.answer)
            for question_answer in chat_history
        ]


@app.post("/query", response_model=APIQueryResponse, status_code=status.HTTP_200_OK)
async def query(
    request: APIQueryRequest,
) -> APIQueryResponse:
    chat_history = get_chat_history(request.chat_history)
    result = chat(
        ChatRequest(question=request.question, chat_history=chat_history),
    )
    result = APIQueryResponse(thread_id=request.thread_id, **result.dict())

    return result


@app.post("/feedback", status_code=status.HTTP_200_OK)
async def feedback(request: APIFeedbackRequest) -> APIFeedbackResponse:
    feedback_response = db_client.update_feedback(request)
    return APIFeedbackResponse(feedback_response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="localhost",
        port=8000,
    )
