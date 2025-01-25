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
from wandbot.models.llm import LLMError

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
    mock_llm_model.create.return_value = EnhancedQuery(**sample_enhanced_query)
    
    result = await query_enhancer({"query": "How do I use wandb?"})
    
    assert result["standalone_query"] == "How do I use wandb?"
    assert not result["avoid_query"]
    assert len(result["intents"]) > 0
    assert len(result["keywords"]) > 0
    assert len(result["sub_queries"]) > 0
    assert len(result["vector_search_queries"]) > 0

@pytest.mark.asyncio
async def test_query_enhancer_validation_error_retry(query_enhancer, mock_llm_model, sample_enhanced_query):
    # Mock validation error then success
    mock_llm_model.create.side_effect = [
        ValidationError.from_exception_data("test", []),
        EnhancedQuery(**sample_enhanced_query)
    ]
    
    result = await query_enhancer({"query": "How do I use wandb?"})
    
    assert result["standalone_query"] == "How do I use wandb?"
    assert mock_llm_model.create.call_count == 2

@pytest.mark.asyncio
async def test_query_enhancer_llm_error(query_enhancer, mock_llm_model):
    # Mock LLM error
    mock_llm_model.create.return_value = LLMError(error=True, error_message="API error")
    
    with pytest.raises(Exception) as exc_info:
        await query_enhancer({"query": "How do I use wandb?"})
    
    assert "API error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_query_enhancer_fallback(query_enhancer, mock_llm_model, sample_enhanced_query):
    # Mock primary model failure and fallback success
    primary_model = mock_llm_model
    fallback_model = AsyncMock()
    
    primary_model.create.side_effect = Exception("Primary failed")
    fallback_model.create.return_value = EnhancedQuery(**sample_enhanced_query)
    
    query_enhancer.fallback_model = fallback_model
    
    result = await query_enhancer({"query": "How do I use wandb?"})
    
    assert result["standalone_query"] == "How do I use wandb?"
    assert primary_model.create.call_count == 3  # We now retry 3 times before fallback
    assert fallback_model.create.call_count == 1 