"""This module provides the setup for the SQLAlchemy database engine and session.

It imports the create_engine and sessionmaker modules from SQLAlchemy, and the DataBaseConfig class from the config
module. It then creates an instance of DataBaseConfig, sets up the engine with the SQLAlchemy database URL and
connection arguments, and creates a sessionmaker bound to this engine.

Typical usage example:

  from wandbot.database.database import SessionLocal
  session = SessionLocal()
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from wandbot.database.config import DataBaseConfig
from wandbot.database.models import Base

db_config = DataBaseConfig()

engine = create_engine(
    db_config.SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
