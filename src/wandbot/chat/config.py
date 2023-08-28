import pathlib

from pydantic import Field
from pydantic_settings import BaseSettings


class ChatConfig(BaseSettings):
    chat_model_name: str = "gpt-4"
    max_retries: int = 2
    fallback_model_name: str = "gpt-3.5-turbo-16k"
    max_fallback_retries: int = 6
    chat_temperature: float = 0.3
    chain_type: str = "stuff"
    chat_prompt: pathlib.Path = pathlib.Path("data/prompts/chat_prompt.json")
    history_prompt: pathlib.Path = pathlib.Path("data/prompts/history_prompt.json")
    retriever_artifact: str = "wandbot/wandbot-dev/vectorstores:latest"
    llm_cache_path: pathlib.Path = pathlib.Path(
        "./data/cache/llm_cache.db", env="LLM_CACHE_PATH"
    )
    verbose: bool = False
    wandb_project: str | None = Field("wandbot-dev", env="WANDB_PROJECT")
    wandb_entity: str | None = Field("wandbot", env="WANDB_ENTITY")
    include_sources: bool = True
    query_tokens_threshold: int = 1024
