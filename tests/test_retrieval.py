from typing import Any, Dict

import pytest

from wandbot.configs.chat_config import ChatConfig
from wandbot.rag.retrieval import FusionRetrievalEngine
from wandbot.retriever.base import VectorStore


@pytest.fixture
def retrieval_engine():
    chat_config = ChatConfig()
    return FusionRetrievalEngine(chat_config=chat_config)

@pytest.mark.asyncio
async def test_retrieval_input_validation(retrieval_engine):
    # Test missing all_queries
    with pytest.raises(ValueError, match="Missing required key 'all_queries'"):
        await retrieval_engine._run_retrieval_common({"standalone_query": "test"}, use_async=True)
    
    # Test missing standalone_query
    with pytest.raises(ValueError, match="Missing required key 'standalone_query'"):
        await retrieval_engine._run_retrieval_common({"all_queries": ["test"]}, use_async=True)
    
    # Test wrong type for all_queries
    with pytest.raises(TypeError, match="Expected 'all_queries' to be a list or tuple"):
        await retrieval_engine._run_retrieval_common(
            {"all_queries": "not a list", "standalone_query": "test"}, 
            use_async=True
        )
    
    # Test wrong type for standalone_query
    with pytest.raises(TypeError, match="Expected 'standalone_query' to be a string"):
        await retrieval_engine._run_retrieval_common(
            {"all_queries": ["test"], "standalone_query": ["not a string"]}, 
            use_async=True
        )
    
    # Test non-dict input
    with pytest.raises(TypeError, match="Expected inputs to be a dictionary"):
        await retrieval_engine._run_retrieval_common("not a dict", use_async=True) 