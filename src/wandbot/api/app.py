import logging
import uuid
from typing import Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import Session
from wandbot.chat_new import Chat, ChatConfig
from wandbot.database import crud, models, schemas
from wandbot.database.database import SessionLocal, engine

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


class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    question_answer_id: str
    sources: Optional[str] = None


def get_chat_history(db: Session, thread_id: str):
    question_answers = crud.get_thread_question_answers(db=db, thread_id=thread_id)
    if question_answers is None or len(question_answers) < 1:
        return []
    else:
        return [(qa.question, qa.answer) for qa in question_answers]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, db: Session = Depends(get_db)):
    question_answer_id = str(uuid.uuid4())
    thread_id = request.thread_id or str(uuid.uuid4())
    chat_history = get_chat_history(db, thread_id)
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
    return QueryResponse(
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
