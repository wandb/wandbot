from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from wandb_utils.stream_table import StreamTable

Base = declarative_base()

#TODO: Validate these are the right columns
class Response(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True)
    source_name = Column(String)
    source_id = Column(String)
    wandb_run_id = Column(String)
    query = Column(String)
    response = Column(String)
    feedback = Column(String)
    elapsed_time = Column(Float)
    start_time = Column(Float)

class DBService:
    def __init__(self, config: dict = {}):
        #TODO: Make this configurable
        DATABASE_URL = "sqlite:///responses.db"
        self.engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
        self.init_db()
        self.wandb_table = self.init_wandb_stream_table("wandbot-results")

    def init_db(self):
        Base.metadata.create_all(bind=self.engine)

    #TODO: Validate these are the right columns
    def init_wandb_stream_table(self, table_name):
        cols = [
            "source_name",
            "source_id",
            "wandb_run_id",
            "query",
            "response",
            "feedback",
            "elapsed_time",
            "start_time",
        ]
        return StreamTable(table_name, cols)

    def log_to_wandb_stream_table(self, source_name, source_id, run_id, query, response, feedback, elapsed_time, start_time):
        self.wandb_table.add_data(source_name, source_id, run_id, query, response, feedback, elapsed_time, start_time)

    def add_response(self, source_name, source_id, wandb_run_id, query, response, feedback, elapsed_time, start_time):
        session = self.SessionLocal()
        try:
            new_response = Response(
                source_name=source_name,
                source_id=source_id,
                wandb_run_id=wandb_run_id,
                query=query,
                response=response,
                feedback=feedback,
                elapsed_time=elapsed_time,
                start_time=start_time,
            )
            session.add(new_response)
            session.commit()
            self.log_to_wandb_stream_table(source_name, source_id, wandb_run_id, query, response, feedback, elapsed_time, start_time)
        finally:
            session.close()

    #TODO: Make this more generic and see if query makes sense here
    def update_feedback(self, source_name, source_id, query, feedback):
        session = self.SessionLocal()
        try:
            response = session.query(Response).filter_by(source_name=source_name, source_id=source_id, query=query).first()
            if response:
                response.feedback = feedback
                session.commit()
        finally:
            session.close()

    def close(self):
        self.SessionLocal.remove()
