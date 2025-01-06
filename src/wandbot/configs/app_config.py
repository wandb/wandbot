from pydantic import Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    wandb_project: str | None = Field("wandbot_public", env="WANDB_PROJECT")
    wandb_entity: str | None = Field("wandbot", env="WANDB_ENTITY")