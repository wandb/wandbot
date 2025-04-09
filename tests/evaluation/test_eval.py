import pytest
import json
from unittest.mock import patch
import httpx

from wandbot.evaluation.eval import (
    get_answer,
    get_record,
    WandbotModel,
    parse_text_to_json
)

# Test data
MOCK_API_RESPONSE = {
    "system_prompt": "test prompt",
    "answer": "test answer",
    "source_documents": "source: https://docs.wandb.ai/test\nThis is a test document",
    "model": "gpt-4",
    "total_tokens": 100,
    "prompt_tokens": 50,
    "completion_tokens": 50,
    "time_taken": 1.5
}

class MockAsyncClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def post(self, *args, **kwargs):
        if self.error:
            raise self.error
        response = self.response
        await response.raise_for_status()
        return response

class MockResponse:
    def __init__(self, data):
        self._data = data
        
    def json(self):
        if isinstance(self._data, dict):
            return self._data
        raise self._data
        
    async def raise_for_status(self):
        if isinstance(self._data, Exception):
            raise self._data

@pytest.mark.asyncio(loop_scope="function")
async def test_get_answer_success():
    """Test successful API call in get_answer."""
    mock_response = MockResponse(MOCK_API_RESPONSE)
    mock_client = MockAsyncClient(response=mock_response)
    
    with patch('httpx.AsyncClient', return_value=mock_client):
        result = await get_answer(
            question="test question",
            wandbot_url="http://test.url",
            application="test-app",
            language="en"
        )
        
        assert json.loads(result) == MOCK_API_RESPONSE

@pytest.mark.asyncio(loop_scope="function")
async def test_get_answer_retry():
    """Test retry behavior in get_answer."""
    attempts = []
    
    class MockClient:
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        async def post(self, *args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.HTTPError("Test error")
            response = MockResponse(MOCK_API_RESPONSE)
            return response
    
    with patch('httpx.AsyncClient', return_value=MockClient()):
        result = await get_answer(
            question="test question",
            wandbot_url="http://test.url"
        )
        
        assert json.loads(result) == MOCK_API_RESPONSE
        assert len(attempts) == 2

@pytest.mark.asyncio(loop_scope="function")
async def test_get_answer_failure():
    """Test complete failure in get_answer."""
    error = httpx.HTTPError("Test error")
    mock_client = MockAsyncClient(error=error)
    
    with patch('httpx.AsyncClient', return_value=mock_client):
        result = await get_answer(
            question="test question",
            wandbot_url="http://test.url"
        )
        
        result_dict = json.loads(result)
        # The error message will contain RetryError due to the retry decorator
        assert "RetryError" in result_dict["error"]
        # Check all other fields match
        assert result_dict["answer"] == ""
        assert result_dict["system_prompt"] == ""
        assert result_dict["source_documents"] == ""
        assert result_dict["model"] == ""
        assert result_dict["total_tokens"] == 0
        assert result_dict["prompt_tokens"] == 0
        assert result_dict["completion_tokens"] == 0
        assert result_dict["time_taken"] == 0

@pytest.mark.asyncio(loop_scope="function")
async def test_get_record_success():
    """Test successful record retrieval."""
    with patch('wandbot.evaluation.eval.get_answer') as mock_get_answer:
        mock_get_answer.return_value = json.dumps(MOCK_API_RESPONSE)
        
        result = await get_record(
            question="test question",
            wandbot_url="http://test.url"
        )
        
        assert result["system_prompt"] == "test prompt"
        assert result["generated_answer"] == "test answer"
        assert len(result["retrieved_contexts"]) == 1
        assert result["retrieved_contexts"][0]["source"] == "https://docs.wandb.ai/test"
        assert not result["has_error"]
        assert result["error_message"] is None

@pytest.mark.asyncio(loop_scope="function")
async def test_get_record_empty_response():
    """Test get_record with empty API response."""
    with patch('wandbot.evaluation.eval.get_answer') as mock_get_answer:
        mock_get_answer.return_value = json.dumps({})
        
        result = await get_record(
            question="test question",
            wandbot_url="http://test.url"
        )
        
        assert result["has_error"]
        assert result["error_message"] == "Unknown API error"
        assert result["generated_answer"] == ""

@pytest.mark.asyncio(loop_scope="function")
async def test_get_record_api_error():
    """Test get_record with API error."""
    with patch('wandbot.evaluation.eval.get_answer') as mock_get_answer:
        mock_get_answer.side_effect = Exception("API Error")
        
        result = await get_record(
            question="test question",
            wandbot_url="http://test.url"
        )
        
        assert result["has_error"]
        assert "Error getting response from wandbotAPI" in result["error_message"]
        assert result["generated_answer"] == ""

def test_parse_text_to_json():
    """Test parsing of source documents text."""
    text = """source: https://docs.wandb.ai/test1
This is document 1
source: https://docs.wandb.ai/test2
This is document 2"""
    
    result = parse_text_to_json(text)
    
    assert len(result) == 2
    assert result[0]["source"] == "https://docs.wandb.ai/test1"
    assert result[0]["content"] == "This is document 1"
    assert result[1]["source"] == "https://docs.wandb.ai/test2"
    assert result[1]["content"] == "This is document 2"

@pytest.mark.asyncio(loop_scope="function")
async def test_wandbot_model():
    """Test WandbotModel prediction."""
    with patch('wandbot.evaluation.eval.get_record') as mock_get_record:
        mock_get_record.return_value = {
            "system_prompt": "test prompt",
            "generated_answer": "test answer",
            "retrieved_contexts": [],
            "model": "gpt-4",
            "total_tokens": 100,
            "has_error": False,
            "error_message": None
        }
        
        model = WandbotModel(
            language="en",
            application="test-app",
            wandbot_url="http://test.url"
        )
        
        result = await model.predict("test question")
        
        assert result["generated_answer"] == "test answer"
        assert not result["has_error"]
        mock_get_record.assert_called_once_with(
            "test question",
            wandbot_url="http://test.url",
            application="test-app",
            language="en"
        ) 