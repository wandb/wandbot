from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", 
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    wandb_project: str | None = Field("wandbot-dev")
    wandb_entity: str | None = Field("wandbot")
    log_level: str | None = Field("INFO")