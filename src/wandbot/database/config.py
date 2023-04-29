from typing import Any

from pydantic import BaseSettings, Field


class DataBaseConfig(BaseSettings):
    SQLALCHEMY_DATABASE_URL: str = Field(
        "sqlite:///./app.db", env="SQLALCHEMY_DATABASE_URL"
    )
    connect_args: dict[str, Any] = Field({"check_same_thread": False})

    class Config:
        env_file = ".env"
