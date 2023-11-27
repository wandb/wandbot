"""Discord bot configuration module.

This module contains the configuration settings for the Discord bot application. 
It includes settings for the application name, wait time, channel IDs, bot token, 
API keys, messages in English and Japanese, API URL, and a flag to include sources.

The settings are defined in the DiscordAppConfig class, which inherits from the 
BaseSettings class provided by the pydantic_settings package. The settings values 
are either hardcoded or fetched from environment variables.

Typical usage example:

  config = DiscordAppConfig()
  wait_time = config.WAIT_TIME
  bot_token = config.DISCORD_BOT_TOKEN
"""

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EN_INTRO_MESSAGE = (
    "🤖 Hi {mention}: \n\n"
    f"Please note that **wandbot** will experience frequent updates.\n\n"
    f"Please do not share any private or sensitive information in your query at this time.\n\n"
    f"Please note that overly long messages (>1024 words) will be truncated!\n\nGenerating response...\n\n"
)

EN_OUTRO_MESSAGE = (
    f"🤖 If you still need help please try re-phrase your question, "
    f"or alternatively reach out to the Weights & Biases Support Team at support@wandb.com \n\n"
    f" Was this response helpful? Please react below to let us know"
)

EN_ERROR_MESSAGE = "Oops!, Sorry 🤖 {mention}: Something went wrong. Please retry again in some time"


class DiscordAppConfig(BaseSettings):
    APPLICATION: str = "Discord"
    WAIT_TIME: float = 300.0
    PROD_DISCORD_CHANNEL_ID: int = 1090739438310654023
    TEST_DISCORD_CHANNEL_ID: int = 1088892013321142484
    DISCORD_BOT_TOKEN: str = Field(..., env="DISCORD_BOT_TOKEN")
    INTRO_MESSAGE: str = Field(EN_INTRO_MESSAGE)
    OUTRO_MESSAGE: str = Field(EN_OUTRO_MESSAGE)
    ERROR_MESSAGE: str = Field(EN_ERROR_MESSAGE)
    WANDBOT_API_URL: AnyHttpUrl = Field(..., env="WANDBOT_API_URL")
    include_sources: bool = True
    bot_language: str = "en"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
