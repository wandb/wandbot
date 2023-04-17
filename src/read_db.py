from services.db_service import DBService, Response

def read_db():
    db = DBService()
    session = db.SessionLocal()
    rows = session.query(Response).all()
    for row in rows:
        print(row.id, row.source_name, row.source_id, row.wandb_run_id, row.query, row.response, row.feedback, row.elapsed_time, row.start_time)

read_db()
