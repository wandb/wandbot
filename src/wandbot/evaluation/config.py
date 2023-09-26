import pathlib

from pydantic import BaseSettings, Field
from wandbot.chat.config import ChatConfig
from wandbot.ingestion.config import VectorIndexConfig


class EvalConfig(BaseSettings):
    debug: bool = True
    eval_model: str = "gpt-3.5-turbo"
    eval_artifact: str = "wandbot/wandbbot/eval_dataset:v0"
    max_retries: int = 3
    retry_delay: int = 10
    eval_prompt: pathlib.Path = pathlib.Path("data/prompts/eval_prompt.txt")
    wandb_project: str = Field(None, env="WANDBOT_WANDB_PROJECT")
    wandb_entity: str | None = Field(None, env="WANDBOT_WANDB_ENTITY")
    wandb_job_type: str | None = Field("eval", env="WANDBOT_WANDB_JOB_TYPE")
    chat_config: ChatConfig = ChatConfig()
    vectorindex_config: VectorIndexConfig = VectorIndexConfig()

    class Config:
        env_prefix = "WANDBOT_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    @root_validator
    def _set_defaults(self, values):
        if values.get("wandb_project") is None:
            values["wandb_entity"] = values["chat_config"].wandb_entity

        return values
