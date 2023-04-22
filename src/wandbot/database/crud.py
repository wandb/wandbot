from sqlalchemy.orm import Session

from . import models, schemas


def get_question_answer(db: Session, question_answer_id: str, thread_id: str):
    return (
        db.query(models.QuestionAnswers)
        .filter(
            models.QuestionAnswers.question_answer_id == question_answer_id,
            models.QuestionAnswers.thread_id == thread_id,
        )
        .first()
    )


def create_question_answer(db: Session, question_answer: schemas.QuestionAnswerCreate):
    db_question_answer = models.QuestionAnswers(
        question_answer_id=question_answer.question_answer_id,
        thread_id=question_answer.thread_id,
        question=question_answer.question,
        answer=question_answer.answer,
        sources=question_answer.sources,
        start_time=question_answer.start_time,
        end_time=question_answer.end_time,
        time_taken=question_answer.time_taken,
    )
    db.add(db_question_answer)
    db.commit()
    db.refresh(db_question_answer)
    return db_question_answer


def get_thread_question_answers(db: Session, thread_id: str):
    return (
        db.query(models.QuestionAnswers)
        .filter(models.QuestionAnswers.thread_id == thread_id)
        .order_by(models.QuestionAnswers.start_time.asc())
        .all()
    )


def update_feedback(db: Session, question_answer: schemas.QuestionAnswer):
    db_question_answer = (
        db.query(models.QuestionAnswers)
        .filter(
            models.QuestionAnswers.question_answer_id
            == question_answer.question_answer_id,
            models.QuestionAnswers.thread_id == question_answer.thread_id,
        )
        .first()
    )
    db_question_answer.feedback = question_answer.feedback
    db.commit()
    db.refresh(db_question_answer)
    return db_question_answer
