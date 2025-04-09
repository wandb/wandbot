from wandbot.configs.chat_config import ChatConfig
from pydantic import ConfigDict

class TestConfig(ChatConfig):
    """Test configuration with minimal retry settings for faster tests"""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"
    )
    
    # Override LLM retry settings
    llm_max_retries: int = 1
    llm_retry_min_wait: int = 1
    llm_retry_max_wait: int = 2
    llm_retry_multiplier: int = 1

    # Override Embedding retry settings
    embedding_max_retries: int = 1
    embedding_retry_min_wait: int = 1
    embedding_retry_max_wait: int = 2
    embedding_retry_multiplier: int = 1

    # Override Reranker retry settings
    reranker_max_retries: int = 1
    reranker_retry_min_wait: int = 1
    reranker_retry_max_wait: int = 2
    reranker_retry_multiplier: int = 1

    # Override retry settings for faster tests
    max_retries: int = 1  # Only try once
    retry_min_wait: int = 1  # Wait 1 second minimum
    retry_max_wait: int = 2  # Wait 2 seconds maximum
    retry_multiplier: int = 1  # No exponential increase 