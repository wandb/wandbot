"""This module contains the configuration settings for wandbot.

The `ChatConfig` class in this module is used to define various settings for wandbot, such as the model name, 
maximum retries, fallback model name, chat temperature, chat prompt, index artifact, embeddings cache, verbosity, 
wandb project and entity, inclusion of sources, and query tokens threshold. These settings are used throughout the 
chatbot's operation to control its behavior.

Typical usage example:

  from wandbot.chat.config import ChatConfig
  config = ChatConfig()
  print(config.chat_model_name)
"""

import pathlib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatConfig(BaseSettings):
    chat_model_name: str = "gpt-4-0125-preview"
    max_retries: int = 2
    fallback_model_name: str = "gpt-3.5-turbo-1106"
    max_fallback_retries: int = 6
    chat_temperature: float = 0.1
    chat_prompt: pathlib.Path = pathlib.Path("data/prompts/chat_prompt.json")
    index_artifact: str = Field(
        "wandbot/wandbot-dev/wandbot_index:latest",
        env="WANDB_INDEX_ARTIFACT",
        validation_alias="wandb_index_artifact",
    )
    embeddings_model: str = "text-embedding-3-small"
    embeddings_dim: int = 512
    verbose: bool = False
    wandb_project: str | None = Field("wandbot_public", env="WANDB_PROJECT")
    wandb_entity: str | None = Field("wandbot", env="WANDB_ENTITY")
    include_sources: bool = True
    query_tokens_threshold: int = 1024

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
