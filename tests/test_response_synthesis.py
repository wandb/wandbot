import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_model_config import TEST_CONFIG, TEST_INVALID_MODELS
from wandbot.models.llm import LLMModel
from wandbot.rag.response_synthesis import ResponseSynthesizer
from wandbot.schema.api_status import APIStatus
from wandbot.schema.document import Document
from wandbot.schema.retrieval import RetrievalResult


@pytest.fixture
def mock_retrieval_result():
    return RetrievalResult(
        documents=[Document(
            page_content="Test content",
            metadata={
                "source": "test_source.md",
                "source_type": "documentation",
                "has_code": False
            }
        )],
        retrieval_info={
            "query": "How do I use wandb?",
            "language": "en",
            "intents": ["test intent"],
            "sub_queries": []
        }
    )

@pytest.fixture
def mock_api_status():
    return APIStatus(
        success=True,
        error_info=None,
        request_id="test_id",
        model_info={"model": "test-model"},
        component="response_synthesis"
    )

@pytest.fixture
def mock_llm_model():
    with patch('wandbot.rag.response_synthesis.LLMModel') as mock:
        instance = mock.return_value
        instance.model_name = "test_model"
        instance.create = AsyncMock()
        yield instance

@pytest.fixture
def synthesizer():
    synth = ResponseSynthesizer(
        primary_provider=TEST_CONFIG["primary"]["provider"],
        primary_model_name=TEST_CONFIG["primary"]["model_name"],
        primary_temperature=TEST_CONFIG["primary"]["temperature"],
        fallback_provider=TEST_CONFIG["fallback"]["provider"],
        fallback_model_name=TEST_CONFIG["fallback"]["model_name"],
        fallback_temperature=TEST_CONFIG["fallback"]["temperature"],
        max_retries=1  # Reduce retries for faster tests
    )
    
    # Mock only the get_messages method to avoid prompt template issues
    synth.get_messages = MagicMock(return_value=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ])
    
    return synth

@pytest.mark.asyncio
async def test_successful_primary_model(synthesizer, mock_retrieval_result):
    """Test when primary model succeeds"""
    result = await synthesizer(mock_retrieval_result)
    
    assert result["response"] is not None
    assert result["response_model"] == TEST_CONFIG["primary"]["model_name"]
    assert "response_synthesis_llm_messages" in result
    assert result["api_statuses"]["response_synthesis_llm_api"].success == True
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0

@pytest.mark.asyncio
async def test_fallback_to_secondary_model(synthesizer, mock_retrieval_result):
    """Test fallback when primary model fails"""
    # Make the primary model fail by using an invalid model name
    synthesizer.model = LLMModel(
        provider=TEST_INVALID_MODELS["primary"]["provider"],
        model_name=TEST_INVALID_MODELS["primary"]["model_name"],
        temperature=TEST_INVALID_MODELS["primary"]["temperature"],
        max_retries=1
    )
    
    result = await synthesizer(mock_retrieval_result)
    
    assert result["response"] is not None
    assert result["response_model"] == TEST_CONFIG["fallback"]["model_name"]
    assert "response_synthesis_llm_messages" in result
    assert result["api_statuses"]["response_synthesis_llm_api"].success == True
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0

@pytest.mark.asyncio
async def test_both_models_fail(synthesizer, mock_retrieval_result):
    """Test behavior when both models fail"""
    # Make both models fail by using invalid model names
    synthesizer.model = LLMModel(
        provider=TEST_INVALID_MODELS["primary"]["provider"],
        model_name=TEST_INVALID_MODELS["primary"]["model_name"],
        temperature=TEST_INVALID_MODELS["primary"]["temperature"],
        max_retries=1
    )
    synthesizer.fallback_model = LLMModel(
        provider=TEST_INVALID_MODELS["fallback"]["provider"],
        model_name=TEST_INVALID_MODELS["fallback"]["model_name"],
        temperature=TEST_INVALID_MODELS["fallback"]["temperature"],
        max_retries=1
    )
    
    with pytest.raises(Exception) as exc_info:
        await synthesizer(mock_retrieval_result)
    
    assert "Response synthesis failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_input_formatting(synthesizer, mock_retrieval_result):
    """Test that inputs are formatted correctly"""
    result = await synthesizer(mock_retrieval_result)
    
    # Check that query and context are formatted correctly
    assert "query_str" in result
    assert "context_str" in result
    assert mock_retrieval_result.retrieval_info["query"] in result["query_str"]
    assert "Test content" in result["context_str"]
    
    # Verify the response is meaningful
    assert result["response"] is not None
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0

if __name__ == "__main__":
    asyncio.run(test_successful_primary_model(synthesizer, mock_retrieval_result)) 