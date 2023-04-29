from sqlalchemy.orm import Session
from wandbot.database import models, schemas


def get_question_answer(
    db: Session, question_answer_id: str, thread_id: str
) -> models.QuestionAnswer:
    return (
        db.query(models.QuestionAnswer)
        .filter(
            models.QuestionAnswer.thread_id == thread_id,
            models.QuestionAnswer.question_answer_id == question_answer_id,
        )
        .first()
    )


def get_chat_thread(db: Session, thread_id: str) -> models.ChatThread:
    return (
        db.query(models.ChatThread)
        .filter(models.ChatThread.thread_id == thread_id)
        .first()
    )


def create_or_update_chat_thread(
    db: Session,
    thread_id: str,
    application: str,
    question_answers: list[schemas.QuestionAnswer] = None,
):
    db_thread = get_chat_thread(db=db, thread_id=thread_id)
    qas = []
    if question_answers:
        for question_answer in question_answers:
            qa = models.QuestionAnswer(**question_answer.dict())
            if not get_question_answer(
                db=db, thread_id=thread_id, question_answer_id=qa.question_answer_id
            ):
                qas.append(qa)
    if db_thread:
        db_thread.question_answers.extend(qas)
        db.flush()
    else:
        db_thread = models.ChatThread(
            thread_id=thread_id,
            application=application,
        )
        db_thread.question_answers.extend(qas)
        db.add(db_thread)
        db.flush()
    db.commit()
    db.refresh(db_thread)
    return db_thread


def update_feedback(db: Session, feedback: schemas.FeedbackBase):
    db_question_answer = (
        db.query(models.QuestionAnswer)
        .filter(
            models.QuestionAnswer.question_answer_id == feedback.question_answer_id,
            models.QuestionAnswer.thread_id == feedback.thread_id,
        )
        .first()
    )
    if db_question_answer is None:
        raise ValueError("Question Answer ID not found")
    db_question_answer.feedback = feedback.feedback
    db.commit()
    db.refresh(db_question_answer)
    return db_question_answer


def get_chat_history(chat_thread: models.ChatThread):
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
