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

from typing import Literal

from pydantic_settings import BaseSettings


class ChatConfig(BaseSettings):    
    # Retrieval settings
    top_k: int = 15
    top_k_per_query: int = 15
    search_type: Literal["mmr", "similarity"] = "mmr"
    do_web_search: bool = False
    redundant_similarity_threshold: float = 0.95  # used to remove very similar retrieved documents
    
    # Retrieval settings: MMR settings
    fetch_k: int = 20  # Used in mmr retrieval. Typically set as top_k * 4
    mmr_lambda_mult: float = 0.5  # used in mmr retrieval
    
    # Reranker models
    rereanker_provider: str = "cohere"
    english_reranker_model: str = "rerank-v3.5"
    multilingual_reranker_model: str = "rerank-v3.5"
    
    # Query enhancer settings
    query_enhancer_provider: str = "google"
    query_enhancer_model: str = "gemini-2.0-flash-001" 
    query_enhancer_temperature: float = 1.0
    query_enhancer_fallback_provider: str = "google"
    query_enhancer_fallback_model: str = "gemini-2.0-flash-001" 
    query_enhancer_fallback_temperature: float = 1.0
    
    # Response synthesis model settings
    response_synthesizer_provider: str = "anthropic"
    response_synthesizer_model: str = "claude-3-5-sonnet-20241022" 
    response_synthesizer_temperature: float = 0.1
    response_synthesizer_fallback_provider: str = "anthropic"
    response_synthesizer_fallback_model: str = "claude-3-5-sonnet-20241022" 
    response_synthesizer_fallback_temperature: float = 0.1
    
    # Translation models settings
    ja_translation_model_name: str = "gpt-4o-2024-08-06"
    
    # LLM Model retry settings
    llm_max_retries: int = 3
    llm_retry_min_wait: float = 4  # minimum seconds to wait between retries
    llm_retry_max_wait: float = 60  # maximum seconds to wait between retries
    llm_retry_multiplier: float = 1  # multiplier for exponential backoff

    # Embedding Model retry settings
    embedding_max_retries: int = 3
    embedding_retry_min_wait: float = 4
    embedding_retry_max_wait: float = 60
    embedding_retry_multiplier: float = 1

    # Reranker retry settings
    reranker_max_retries: int = 5
    reranker_retry_min_wait: float = 2.0
    reranker_retry_max_wait: float = 180
    reranker_retry_multiplier: float = 2.5

    # Vector Store retry settings
    vector_store_max_retries: int = 3
    vector_store_retry_min_wait: float = 1.0  # Start with a short wait
    vector_store_retry_max_wait: float = 10.0 # Cap the wait time
    vector_store_retry_multiplier: float = 2.0 # Double wait time each retry
