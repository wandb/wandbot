"""This module contains the configuration settings for the Slack application.

This module uses the Pydantic library to define the configuration settings for the Slack application. 
These settings include tokens, secrets, API keys, and messages for the application. 
The settings are loaded from an environment file and can be accessed as properties of the `SlackAppEnConfig` class.

Typical usage example:

  from .config import SlackAppEnConfig

  config = SlackAppEnConfig()
  token = config.SLACK_APP_TOKEN
"""

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EN_INTRO_MESSAGE = (
    "Hi <@{user}>:\n\n"
    f"Please note that *wandbot is currently in alpha testing* and will experience frequent updates.\n\n"
    f"Please do not share any private or sensitive information in your query at this time.\n\n"
    f"Please note that overly long messages (>1024 words) will be truncated!\n\nGenerating response...\n\n"
)

EN_OUTRO_MESSAGE = (
    f"🤖 If you still need help please try re-phrase your question, "
    f"or alternatively reach out to the Weights & Biases Support Team at support@wandb.com \n\n"
    f" Was this response helpful? Please react below to let us know"
)

EN_ERROR_MESSAGE = (
    "Oops!, Something went wrong. Please retry again in some time"
)

EN_FALLBACK_WARNING_MESSAGE = (
    "*Warning: Falling back to {model}*, These results may nor be as good as "
    "*gpt-4*\n\n"
)

JA_INTRO_MESSAGE = (
    "こんにちは <@{user}>:\n\n"
    "Wandbotは現在アルファテスト中ですので、頻繁にアップデートされます。"
    "ご利用の際にはプライバシーに関わる情報は入力されないようお願いします。返答を生成しています・・・"
)

JA_OUTRO_MESSAGE = (
    ":robot_face: この答えが十分でなかった場合には、質問を少し変えて試してみると結果が良くなることがあるので、お試しください。もしくは、"
    "#support チャンネルにいるwandbチームに質問してください。この答えは役に立ったでしょうか？下のボタンでお知らせ下さい。"
)

JA_ERROR_MESSAGE = "「おっと、問題が発生しました。しばらくしてからもう一度お試しください。」"

JA_FALLBACK_WARNING_MESSAGE = (
    "*警告: {model}* にフォールバックします。これらの結果は *gpt-4* ほど良くない可能性があります*\n\n"
)


class SlackAppEnConfig(BaseSettings):
    APPLICATION: str = Field("Slack")
    SLACK_APP_TOKEN: str = Field(..., validation_alias="SLACK_EN_APP_TOKEN")
    SLACK_BOT_TOKEN: str = Field(..., validation_alias="SLACK_EN_BOT_TOKEN")
    SLACK_SIGNING_SECRET: str = Field(
        ..., validation_alias="SLACK_EN_SIGNING_SECRET"
    )
    WANDB_API_KEY: str = Field(..., validation_alias="WANDB_API_KEY")
    INTRO_MESSAGE: str = Field(EN_INTRO_MESSAGE)
    OUTRO_MESSAGE: str = Field(EN_OUTRO_MESSAGE)
    ERROR_MESSAGE: str = Field(EN_ERROR_MESSAGE)
    WARNING_MESSAGE: str = Field(EN_FALLBACK_WARNING_MESSAGE)
    WANDBOT_API_URL: AnyHttpUrl = Field(..., validation_alias="WANDBOT_API_URL")
    include_sources: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    lang_code: str = "en"


class SlackAppJaConfig(BaseSettings):
    APPLICATION: str = Field("Slack")
    SLACK_APP_TOKEN: str = Field(..., validation_alias="SLACK_JA_APP_TOKEN")
    SLACK_BOT_TOKEN: str = Field(..., validation_alias="SLACK_JA_BOT_TOKEN")
    SLACK_SIGNING_SECRET: str = Field(
        ..., validation_alias="SLACK_JA_SIGNING_SECRET"
    )
    WANDB_API_KEY: str = Field(..., validation_alias="WANDB_API_KEY")
    INTRO_MESSAGE: str = Field(JA_INTRO_MESSAGE)
    OUTRO_MESSAGE: str = Field(JA_OUTRO_MESSAGE)
    ERROR_MESSAGE: str = Field(JA_ERROR_MESSAGE)
    WARNING_MESSAGE: str = Field(JA_FALLBACK_WARNING_MESSAGE)
    WANDBOT_API_URL: AnyHttpUrl = Field(..., validation_alias="WANDBOT_API_URL")
    include_sources: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    lang_code: str = "ja"
