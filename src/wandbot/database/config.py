from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class DataBaseConfig(BaseSettings):
    SQLALCHEMY_DATABASE_URL: str = Field("sqlite:///./data/cache/app.db", env="SQLALCHEMY_DATABASE_URL")
    connect_args: dict[str, Any] = Field({"check_same_thread": False})
