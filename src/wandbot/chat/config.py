import pathlib

from pydantic import Field
from pydantic_settings import BaseSettings


class ChatConfig(BaseSettings):
    chat_model_name: str = "gpt-4-0613"
    max_retries: int = 2
    fallback_model_name: str = "gpt-3.5-turbo-0613"
    max_fallback_retries: int = 6
    chat_temperature: float = 0.0
    chat_prompt: pathlib.Path = pathlib.Path("data/prompts/chat_prompt.json")
    index_artifact: str = "wandbot/wandbot-dev/wandbot_index:latest"
    embeddings_cache: pathlib.Path = Field(pathlib.Path("./data/cache/embeddings"), env="EMBEDDINGS_CACHE_PATH")
    verbose: bool = False
    wandb_project: str | None = Field("wandbot_public", env="WANDB_PROJECT")
    wandb_entity: str | None = Field("wandbot", env="WANDB_ENTITY")
    include_sources: bool = True
    query_tokens_threshold: int = 1024
