from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from wandbot.database.config import DataBaseConfig

db_config = DataBaseConfig()

engine = create_engine(
    db_config.SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
