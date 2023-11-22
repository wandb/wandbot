from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ZDGROUPID = "360016040851"


class ZendeskAppConfig(BaseSettings):
    ZENDESK_EMAIL: str = (Field(..., env="ZENDESK_EMAIL"),)
    ZENDESK_PASSWORD: str = (Field(..., env="ZENDESK_PASSWORD"),)
    ZENDESK_SUBDOMAIN: str = (Field(..., env="ZENDESK_SUBDOMAIN"),)

    WANDB_API_KEY: str = Field(..., env="WANDB_API_KEY")
    ZDGROUPID: str = ZDGROUPID
    WANDBOT_API_URL: AnyHttpUrl = Field(..., env="WANDBOT_API_URL")
    include_sources: bool = True
    bot_language: str = "en"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
