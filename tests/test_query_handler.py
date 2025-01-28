import pytest
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError

from wandbot.rag.query_handler import (
    QueryEnhancer, 
    EnhancedQuery, 
    clean_question,
    format_chat_history,
    Labels
)
from wandbot.schema.api_status import APIStatus, ErrorInfo

@pytest.fixture
def mock_llm_model():
    with patch('wandbot.rag.query_handler.LLMModel') as mock:
        instance = mock.return_value
        instance.create = AsyncMock()
        yield instance

@pytest.fixture
def query_enhancer(mock_llm_model):
    return QueryEnhancer(
        model_name="gpt-4",
        temperature=0,
        fallback_model_name="gpt-3.5-turbo",
        fallback_temperature=0
    )

@pytest.fixture
def sample_enhanced_query():
    return {
        "language": "en",
        "intents": [
            {
                "reasoning": "The query is about using W&B features",
                "label": Labels.PRODUCT_FEATURES
            }
        ],
        "keywords": [
            {"keyword": "wandb features"}
        ],
        "sub_queries": [
            {"query": "How to use wandb?"}
        ],
        "vector_search_queries": [
            {"query": "wandb basic usage"}
        ],
        "standalone_query": "How do I use wandb?"
    }

def test_clean_question():
    assert clean_question("<@U123456> hello") == "hello"
    assert clean_question("@bot how are you?") == "how are you?"
    assert clean_question("regular question") == "regular question"

def test_format_chat_history():
    history = [
        ("Hello", "Hi there!"),
        ("How are you?", "I'm good!")
    ]
    formatted = format_chat_history(history)
    assert "User: Hello" in formatted
    assert "Assistant: Hi there!" in formatted
    assert "User: How are you?" in formatted
    assert "Assistant: I'm good!" in formatted

    assert format_chat_history(None) == "No chat history available."
    assert format_chat_history([]) == "No chat history available."

@pytest.mark.asyncio
async def test_query_enhancer_success(query_enhancer, mock_llm_model, sample_enhanced_query):
    # Mock successful response
    enhanced_query = EnhancedQuery(**sample_enhanced_query)
    api_status = APIStatus(component="llm", success=True)
    mock_llm_model.create.return_value = (enhanced_query, api_status)
    
    result = await query_enhancer({"query": "How do I use wandb?"})
    
    assert result is not None
    assert result["query_enhancer_success"]
    assert result["query_enhancer_error"] is None

@pytest.mark.asyncio
async def test_query_enhancer_validation_error_retry(query_enhancer, mock_llm_model, sample_enhanced_query):
    # Mock validation error then success
    enhanced_query = EnhancedQuery(**sample_enhanced_query)
    error_status = APIStatus(component="llm", success=False, error_info=ErrorInfo(has_error=True, error_message="Validation error"))
    success_status = APIStatus(component="llm", success=True)
    
    mock_llm_model.create.side_effect = [
        (None, error_status),
        (enhanced_query, success_status)
    ]
    
    result = await query_enhancer({"query": "How do I use wandb?"})
    
    assert result is not None
    assert result["query_enhancer_success"]
    assert result["query_enhancer_error"] is None

@pytest.mark.asyncio
async def test_query_enhancer_llm_error(query_enhancer, mock_llm_model):
    # Mock LLM error that persists through retries
    error_status = APIStatus(component="llm", success=False, error_info=ErrorInfo(has_error=True, error_message="API error"))
    mock_llm_model.create.return_value = (None, error_status)
    
    # Mock the fallback model to also fail
    query_enhancer.fallback_model.create.return_value = (None, error_status)
    
    with pytest.raises(Exception) as exc_info:
        await query_enhancer({"query": "How do I use wandb?"})
    
    # The error will be wrapped in a retry error message
    assert "API error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_query_enhancer_fallback(query_enhancer, mock_llm_model, sample_enhanced_query):
    # Mock primary model failure and fallback success
    primary_model = mock_llm_model
    fallback_model = AsyncMock()
    
    enhanced_query = EnhancedQuery(**sample_enhanced_query)
    error_status = APIStatus(component="llm", success=False, error_info=ErrorInfo(has_error=True, error_message="Primary failed"))
    success_status = APIStatus(component="llm", success=True)
    
    primary_model.create.return_value = (None, error_status)
    fallback_model.create.return_value = (enhanced_query, success_status)
    
    query_enhancer.fallback_model = fallback_model
    
    result = await query_enhancer({"query": "How do I use wandb?"})
    
    assert result is not None
    assert result["query_enhancer_success"]
    assert result["query_enhancer_error"] is None 