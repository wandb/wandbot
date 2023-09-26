from pydantic import AnyHttpUrl, BaseSettings, Field

INTRO_MESSAGE = (
    f"Please note that **wandbot is currently in alpha testing** and will experience frequent updates.\n\n"
    f"Please do not share any private or sensitive information in your query at this time.\n\n"
    f"Please note that overly long messages (>1024 words) will be truncated!\n\nGenerating response...\n\n"
)

OUTRO_MESSAGE = (
    f"🤖 If you still need help please try re-phrase your question, "
    f"or alternatively reach out to the Weights & Biases Support Team at support@wandb.com \n\n"
    f" Was this response helpful? Please react below to let us know"
)

ERROR_MESSAGE = "Oops!, Something went wrong. Please retry again in some time"


class DiscordAppConfig(BaseSettings):
    APPLICATION: str = "Discord"
    WAIT_TIME: float = 300.0
    PROD_DISCORD_CHANNEL_ID: int = 1090739438310654023
    TEST_DISCORD_CHANNEL_ID: int = 1088892013321142484
    DISCORD_BOT_TOKEN: str = Field(..., env="DISCORD_BOT_TOKEN")
    WANDB_API_KEY: str = Field(..., env="WANDB_API_KEY")
    INTRO_MESSAGE: str = Field(INTRO_MESSAGE)
    OUTRO_MESSAGE: str = Field(OUTRO_MESSAGE)
    ERROR_MESSAGE: str = Field(ERROR_MESSAGE)
    WANDBOT_API_URL: AnyHttpUrl = Field(..., env="WANDBOT_API_URL")
    include_sources: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
