from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings

EN_INTRO_MESSAGE = (
    f"Please note that **wandbot is currently in alpha testing** and will experience frequent updates.\n\n"
    f"Please do not share any private or sensitive information in your query at this time.\n\n"
    f"Please note that overly long messages (>1024 words) will be truncated!\n\nGenerating response...\n\n"
)

EN_OUTRO_MESSAGE = (
    f"🤖 If you still need help please try re-phrase your question, "
    f"or alternatively reach out to the Weights & Biases Support Team at support@wandb.com \n\n"
    f" Was this response helpful? Please react below to let us know"
)

EN_ERROR_MESSAGE = "Oops!, Something went wrong. Please retry again in some time"

JA_INTRO_MESSAGE = (
    "Wandbotは現在アルファテスト中ですので、頻繁にアップデートされます。"
    "ご利用の際にはプライバシーに関わる情報は入力されないようお願いします。返答を生成しています・・・"
)

JA_OUTRO_MESSAGE = (
    ":robot_face: この答えが十分でなかった場合には、質問を少し変えて試してみると結果が良くなることがあるので、お試しください。もしくは、"
    "#support チャンネルにいるwandbチームに質問してください。この答えは役に立ったでしょうか？下のボタンでお知らせ下さい。"
)

JA_ERROR_MESSAGE = "「おっと、問題が発生しました。しばらくしてからもう一度お試しください。」"


class DiscordAppConfig(BaseSettings):
    APPLICATION: str = "Discord"
    WAIT_TIME: float = 300.0
    PROD_DISCORD_CHANNEL_ID: int = 1090739438310654023
    TEST_DISCORD_CHANNEL_ID: int = 1088892013321142484
    DISCORD_BOT_TOKEN: str = Field(..., env="DISCORD_BOT_TOKEN")
    WANDB_API_KEY: str = Field(..., env="WANDB_API_KEY")
    EN_INTRO_MESSAGE: str = Field(EN_INTRO_MESSAGE)
    EN_OUTRO_MESSAGE: str = Field(EN_OUTRO_MESSAGE)
    EN_ERROR_MESSAGE: str = Field(EN_ERROR_MESSAGE)
    JA_INTRO_MESSAGE: str = Field(JA_INTRO_MESSAGE)
    JA_OUTRO_MESSAGE: str = Field(JA_OUTRO_MESSAGE)
    JA_ERROR_MESSAGE: str = Field(JA_ERROR_MESSAGE)
    WANDBOT_API_URL: AnyHttpUrl = Field(..., env="WANDBOT_API_URL")
    include_sources: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
