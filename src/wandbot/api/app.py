import logging
import uuid
from typing import Optional

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from wandbot.api import crud, models, schemas
from wandbot.api.database import SessionLocal, engine
from wandbot.api.schemas import APIQueryRequest, APIQueryResponse
from wandbot.chat import Chat, ChatConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)
chat: Optional[Chat] = None
app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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


@app.post("/query", response_model=APIQueryResponse)
async def query(request: APIQueryRequest, db: Session = Depends(get_db)):
    question_answer_id = str(uuid.uuid4())
    thread_id = request.thread_id or str(uuid.uuid4())
    chat_history = get_chat_history(db, thread_id)
    if not chat_history:
        crud.create_chat_thread(
            db=db, thread_id=thread_id, application=request.application
        )
    logger.info(f"chat_history: {chat_history}")
    result = chat(question=request.question, chat_history=chat_history)
    question_answer = schemas.QuestionAnswerCreate(
        question_answer_id=question_answer_id,
        thread_id=thread_id,
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        start_time=result["start_time"],
        end_time=result["end_time"],
        time_taken=result["time_taken"],
    )
    db_response = crud.create_question_answer(db=db, question_answer=question_answer)
    return APIQueryResponse(
        answer=db_response.answer,
        sources=db_response.sources,
        thread_id=db_response.thread_id,
        question_answer_id=db_response.question_answer_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
