import pathlib

from pydantic import BaseSettings, Field, root_validator
from wandbot.ingestion.config import VectorIndexConfig


class ChatConfig(BaseSettings):
    model_name: str = "gpt-4"
    max_retries: int = 1
    fallback_model_name: str = "gpt-3.5-turbo"
    max_fallback_retries: int = 6
    chat_temperature: float = 0.0
    chain_type: str = "map_reduce"
    chat_prompt: pathlib.Path = pathlib.Path("data/prompts/chat_prompt.json")
    history_prompt: pathlib.Path = pathlib.Path("data/prompts/history_prompt.json")
    vectorindex_config: VectorIndexConfig = VectorIndexConfig(
        wandb_project="wandb_docs_bot_dev",  # TODO: change this to the correct project using ENV
    )
    vectorindex_artifact: str = (
        "parambharat/wandb_docs_bot_dev/wandbot_vectorindex:latest"
    )
    llm_cache_path: pathlib.Path = pathlib.Path("llm_cache.db", env="LLM_CACHE_PATH")
    verbose: bool = False
    wandb_project: str | None = Field(None, env="WANDBOT_WANDB_PROJECT")
    wandb_entity: str | None = Field(None, env="WANDBOT_WANDB_ENTITY")
    wandb_job_type: str | None = "chat"
    include_sources: bool = True
    source_score_threshold: float = 1.0
    query_tokens_threshold: int = 1024

    class Config:
        env_prefix = "WANDBOT_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    @root_validator(pre=False)
    def _set_defaults(cls, values):
        if values["wandb_project"] is None:
            values["wandb_project"] = values["vectorindex_config"].wandb_project
        if values["wandb_entity"] is None:
            values["wandb_entity"] = values["vectorindex_config"].wandb_entity
        return values
