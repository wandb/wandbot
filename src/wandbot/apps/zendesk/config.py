from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ZDGROUPID = "360016040851"


class zendesk_app_config(BaseSettings):
    DISCBOTINTRO: str = (
        "Thank you for reaching out to W&B Technical Support!\n\n"
    f"This is an automated reply from our support bot designed to assist you with your WandB-related queries.\n"
    f"If you find the solution unsatisfactory or have additional questions, we encourage you to contact our support team at support@wandb.com, or continue replying in this thread\n\n"
    )
    
    ZENDESK_EMAIL: str = (Field(..., env="ZENDESK_EMAIL"),)
    ZENDESK_PASSWORD: str = (Field(..., env="ZENDESK_PASSWORD"),)
    ZENDESK_SUBDOMAIN: str = (Field(..., env="ZENDESK_SUBDOMAIN"),)
    ZDGROUPID: str = ZDGROUPID
    WANDBOT_API_URL: AnyHttpUrl = Field(..., env="WANDBOT_API_URL")
    MAX_WANDBOT_REQUESTS: int = 5
    REQUEST_INTERVAL: int = 60
    INTERVAL_TO_FETCH_TICKETS: int = 600
    include_sources: bool = True
    bot_language: str = "en"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
