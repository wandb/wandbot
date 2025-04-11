"""This module provides the setup for the SQLAlchemy database engine and session.

It imports the create_engine and sessionmaker modules from SQLAlchemy, and the DataBaseConfig class from the config
module. It then creates an instance of DataBaseConfig, sets up the engine with the SQLAlchemy database URL and
connection arguments, and creates a sessionmaker bound to this engine.

Typical usage example:

  from wandbot.database.database import SessionLocal
  session = SessionLocal()
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from wandbot.configs.database_config import DataBaseConfig

db_config = DataBaseConfig()

# Ensure the directory for the SQLite database exists
db_url = db_config.SQLALCHEMY_DATABASE_URL
if db_url.startswith("sqlite:///"):
    db_path_str = db_url.split("sqlite:///", 1)[1]
    db_path = Path(db_path_str)
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    db_url, connect_args=db_config.connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
