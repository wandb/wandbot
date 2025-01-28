"""This module provides a DataBaseConfig class for managing database configuration.

The DataBaseConfig class uses the BaseSettings class from pydantic_settings to define and manage the database
configuration settings. It includes the SQLAlchemy database URL and connection arguments.

Typical usage example:

  db_config = DataBaseConfig()
  database_url = db_config.SQLALCHEMY_DATABASE_URL
  connect_args = db_config.connect_args
"""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataBaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", env_file_encoding="utf-8")
    
    SQLALCHEMY_DATABASE_URL: str = Field(
        "sqlite:///./data/cache/app.db"
    )
    connect_args: dict[str, Any] = Field({"check_same_thread": False})
