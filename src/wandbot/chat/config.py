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

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    index_artifact: str = Field(
        "wandbot/wandbot_public/wandbot_chroma_index:v0",
        env="WANDB_INDEX_ARTIFACT",
        validation_alias="wandb_index_artifact",
    )
    wandb_project: str | None = Field("wandbot_public", env="WANDB_PROJECT")
    wandb_entity: str | None = Field("wandbot", env="WANDB_ENTITY")
    # Retrieval settings
    top_k: int = 15
    search_type: str = "mmr"
    # Cohere reranker models
    english_reranker_model: str = "rerank-english-v2.0"
    multilingual_reranker_model: str = "rerank-multilingual-v2.0"
    # Response synthesis settings
    response_synthesizer_model: str = "gpt-4-0125-preview"
    response_synthesizer_temperature: float = 0.1
    response_synthesizer_fallback_model: str = "gpt-4-0125-preview"
    response_synthesizer_fallback_temperature: float = 0.1
