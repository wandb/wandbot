"""This module contains the configuration settings for wandbot.

The `ChatConfig` class in this module is used to define various settings for wandbot, such as the model name, 
maximum retries, fallback model name, chat temperature, chat prompt, index artifact, embeddings cache, verbosity, 
wandb project and entity, inclusion of sources, and query tokens threshold. These settings are used throughout the 
chatbot's operation to control its behavior.

Typical usage example:

  from wandbot.configs.chat_config import ChatConfig
  config = ChatConfig()
  print(config.chat_model_name)
"""

from pydantic_settings import BaseSettings


class ChatConfig(BaseSettings):
    # Retrieval settings
    top_k: int = 15
    top_k_per_query: int = 15
    search_type: str = "mmr"
    do_web_search: bool = False
    redundant_similarity_threshold: float = 0.95  # used to remove very similar retrieved documents
    similarity_score_threshold: float = 0.0  # Used in similarity score threshold retrieval, similarity search used
    # Retrieval settings: MMR settings
    fetch_k: int = 20  # Used in mmr retrieval. Typically set as top_k * 4
    mmr_lambda_mult: float = 0.5  # used in mmr retrieval
    # Reranker models
    rereanker_provider: str = "cohere"
    english_reranker_model: str = "rerank-english-v2.0"
    multilingual_reranker_model: str = "rerank-multilingual-v2.0"
    # Query enhancer settings
    query_enhancer_model: str = "gpt-4-0125-preview"
    query_enhancer_temperature: float = 0.1
    query_enhancer_fallback_model: str = "gpt-4-0125-preview"
    query_enhancer_fallback_temperature: float = 0.1
    # Response synthesis model settings
    response_synthesizer_provider: str = "openai"
    response_synthesizer_model: str = "gpt-4-0125-preview"
    response_synthesizer_temperature: float = 0.1
    response_synthesizer_fallback_provider: str = "openai"
    response_synthesizer_fallback_model: str = "gpt-4-0125-preview"
    response_synthesizer_fallback_temperature: float = 0.1
    # Translation models settings
    ja_translation_model_name: str = "gpt-4o-2024-08-06"
