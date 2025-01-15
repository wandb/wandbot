from pydantic import Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    wandb_project: str | None = Field("wandbot-dev", env="WANDB_PROJECT")
    wandb_entity: str | None = Field("wandbot", env="WANDB_ENTITY")
    log_level: str | None = Field("INFO", env="LOG_LEVEL")