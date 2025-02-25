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
    f"Please note that **wandbot** will experience frequent updates.\n\n"
    f"Please do not share any private or sensitive information in your query.\n\n"
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
    "**Warning: Falling back to {model}**, These results may nor be as good as "
    "**gpt-4**\n\n"
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
    "**警告: {model}** にフォールバックします。これらの結果は **gpt-4** ほど良くない可能性があります\n\n"
)

KO_INTRO_MESSAGE = (
    "안녕하세요 <@{user}>:\n\n"
    "Wandbot는 현재 알파 테스트 중이며, 자주 업데이트됩니다. \n"
    "사용 시 개인 정보를 입력하지 않도록 주의해주십시오. 답변을 생성하고 있습니다... "
)


KO_OUTRO_MESSAGE = (
    "🤖 여전히 도움이 필요하시면 질문을 조금 바꿔서 다시 시도해 보시거나, "
    "또는 Weights & Biases 지원팀(support@wandb.com)에 문의해 주세요. \n\n"
    "이 답변이 도움이 되었나요? 아래 버튼을 눌러 알려주세요!"
)

KO_ERROR_MESSAGE = "「오류가 발생했습니다. 잠시 후 다시 시도해 주세요.」"

KO_FALLBACK_WARNING_MESSAGE = (
    "**경고: {model}** 에 폴백합니다. 이 결과는 **gpt-4** 보다 나을 수도 있지만 좋지 않을 수도 있습니다.\n\n"
)


class SlackAppEnConfig(BaseSettings):
    APPLICATION: str = Field("Slack_EN")
    SLACK_APP_TOKEN: str = Field(..., validation_alias="SLACK_EN_APP_TOKEN")
    SLACK_BOT_TOKEN: str = Field(..., validation_alias="SLACK_EN_BOT_TOKEN")
    SLACK_SIGNING_SECRET: str = Field(
        ..., validation_alias="SLACK_EN_SIGNING_SECRET"
    )
    INTRO_MESSAGE: str = Field(EN_INTRO_MESSAGE)
    OUTRO_MESSAGE: str = Field(EN_OUTRO_MESSAGE)
    ERROR_MESSAGE: str = Field(EN_ERROR_MESSAGE)
    WARNING_MESSAGE: str = Field(EN_FALLBACK_WARNING_MESSAGE)
    WANDBOT_API_URL: AnyHttpUrl = Field(..., validation_alias="WANDBOT_API_URL")
    include_sources: bool = True
    bot_language: str = "en"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


class SlackAppJaConfig(BaseSettings):
    APPLICATION: str = Field("Slack_JA")
    SLACK_APP_TOKEN: str = Field(..., validation_alias="SLACK_JA_APP_TOKEN")
    SLACK_BOT_TOKEN: str = Field(..., validation_alias="SLACK_JA_BOT_TOKEN")
    SLACK_SIGNING_SECRET: str = Field(
        ..., validation_alias="SLACK_JA_SIGNING_SECRET"
    )
    INTRO_MESSAGE: str = Field(JA_INTRO_MESSAGE)
    OUTRO_MESSAGE: str = Field(JA_OUTRO_MESSAGE)
    ERROR_MESSAGE: str = Field(JA_ERROR_MESSAGE)
    WARNING_MESSAGE: str = Field(JA_FALLBACK_WARNING_MESSAGE)
    WANDBOT_API_URL: AnyHttpUrl = Field(..., validation_alias="WANDBOT_API_URL")
    include_sources: bool = True
    bot_language: str = "ja"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


class SlackAppKoConfig(BaseSettings):
    APPLICATION: str = Field("Slack_KO")
    SLACK_APP_TOKEN: str = Field(..., validation_alias="SLACK_KO_APP_TOKEN")
    SLACK_BOT_TOKEN: str = Field(..., validation_alias="SLACK_KO_BOT_TOKEN")
    SLACK_SIGNING_SECRET: str = Field(
        ..., validation_alias="SLACK_KO_SIGNING_SECRET"
    )
    INTRO_MESSAGE: str = Field(KO_INTRO_MESSAGE)
    OUTRO_MESSAGE: str = Field(KO_OUTRO_MESSAGE)
    ERROR_MESSAGE: str = Field(KO_ERROR_MESSAGE)
    WARNING_MESSAGE: str = Field(KO_FALLBACK_WARNING_MESSAGE)
    WANDBOT_API_URL: AnyHttpUrl = Field(..., validation_alias="WANDBOT_API_URL")
    include_sources: bool = True
    bot_language: str = "ko"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
