import logging
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Request, Response, status
from sqlalchemy.orm import Session
from wandbot.api import crud, models, schemas
from wandbot.api.database import SessionLocal, engine
from wandbot.api.schemas import APIQueryRequest, APIQueryResponse
from wandbot.chat import Chat
from wandbot.config import ChatConfig, ChatRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)
chat: Optional[Chat] = None
app = FastAPI(name="wandbot", version="0.0.1")


# Dependency
@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    response = Response("Internal server error", status_code=500)
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        request.state.db.close()
    return response


# Dependency
def get_db(request: Request):
    return request.state.db


@app.on_event("startup")
def startup_event():
    global chat
    chat = Chat(ChatConfig())


def get_chat_history(db: Session, thread_id: str):
    chat_thread = crud.get_thread(db=db, thread_id=thread_id)
    if chat_thread is not None:
        if (
            chat_thread.question_answers is None
            or len(chat_thread.question_answers) < 1
        ):
            result = []
        else:
            result = [(qa.question, qa.answer) for qa in chat_thread.question_answers]
    else:
        result = []
    return result


@app.post("/query", response_model=APIQueryResponse, status_code=status.HTTP_200_OK)
async def query(request: APIQueryRequest, db: Session = Depends(get_db)):
    question_answer_id = request.question_answer_id or str(uuid.uuid4())
    thread_id = request.thread_id or str(uuid.uuid4())
    chat_history = get_chat_history(db, thread_id)
    if not chat_history:
        crud.create_chat_thread(
            db=db, thread_id=thread_id, application=request.application
        )
    result = chat(ChatRequest(question=request.question, chat_history=chat_history))
    question_answer = schemas.QuestionAnswerCreate(
        question_answer_id=question_answer_id,
        thread_id=thread_id,
        **result.dict(),
    )
    db_response = crud.create_question_answer(db=db, question_answer=question_answer)
    if result.model != chat.config.model_name:
        answer = (
            f"**Warning: Falling back to {chat.config.fallback_model_name}.** These results are "
            f"sometimes not as good as {chat.config.model_name} \n\n"
            + db_response.answer
        )
    else:
        answer = db_response.answer
    return APIQueryResponse(
        answer=answer,
        sources=db_response.sources,
        source_documents=db_response.source_documents,
        thread_id=db_response.thread_id,
        question_answer_id=db_response.question_answer_id,
    )


@app.post("/feedback", status_code=status.HTTP_201_CREATED)
async def feedback(request: schemas.APIFeedbackRequest, db: Session = Depends(get_db)):
    crud.update_feedback(db=db, feedback=request)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
