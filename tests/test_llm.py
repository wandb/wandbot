import pytest
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio

from wandbot.models.llm import (
    AsyncOpenAILLMModel,
    AsyncAnthropicLLMModel,
    LLMModel,
    extract_system_and_messages,
)
from wandbot.utils import ErrorInfo
from wandbot.schema.api_status import APIStatus

# Load environment variables from .env
load_dotenv()

class ResponseModelForTest(BaseModel):
    answer: str
    confidence: float

class SimpleResponse(BaseModel):
    answer: int

anthropic_models = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022"
]

openai_models = [
    "gpt-4-1106-preview",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "o1-2024-12-17",
    "o1-mini-2024-09-12"
]

@pytest.fixture
def messages():
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

@pytest.fixture
def response_model():
    return ResponseModelForTest

# Utility function tests
def test_extract_system_and_messages(messages):
    system_msg, chat_msgs = extract_system_and_messages(messages)
    assert system_msg == "You are a helpful assistant"
    assert len(chat_msgs) == 2
    assert all(msg["role"] in ["user", "assistant"] for msg in chat_msgs)

def test_extract_system_and_messages_no_system():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    system_msg, chat_msgs = extract_system_and_messages(msgs)
    assert system_msg is None
    assert len(chat_msgs) == 2

# OpenAI model tests
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", openai_models)
async def test_openai_llm_creation(model_name):
    model = AsyncOpenAILLMModel(model_name=model_name, temperature=0)
    assert model.model_name == model_name
    assert model.temperature == 0
    assert model.client.api_key == os.getenv("OPENAI_API_KEY")

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", openai_models)
async def test_openai_llm_create(model_name):
    model = AsyncOpenAILLMModel(model_name=model_name, temperature=0)
    result, api_status = await model.create([
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ])
    assert result.strip() == "4"
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "openai"

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", openai_models)
async def test_openai_llm_create_with_response_model(model_name):
    model = AsyncOpenAILLMModel(
        model_name=model_name, 
        temperature=0,
        response_model=SimpleResponse
    )
    result, api_status = await model.create([
        {"role": "user", "content": "Return the number 4 as a JSON object with the key 'answer'. Respond with only valid JSON."}
    ])
    
    assert isinstance(result, SimpleResponse)
    assert result.answer == 4
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "openai"

@pytest.mark.asyncio
async def test_openai_invalid_model():
    client = AsyncOpenAILLMModel(model_name="invalid-model")
    result, api_status = await client.create([{"role": "user", "content": "test"}])
    assert result is None
    assert isinstance(api_status, APIStatus)
    assert not api_status.success
    assert api_status.error_info is not None
    assert "model_not_found" in api_status.error_info.error_message
    assert api_status.component == "openai"

# Anthropic model tests
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", anthropic_models)
async def test_anthropic_llm_creation(model_name):
    model = AsyncAnthropicLLMModel(model_name=model_name, temperature=0)
    assert model.model_name == model_name
    assert model.temperature == 0
    assert model.client.api_key == os.getenv("ANTHROPIC_API_KEY")

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", anthropic_models)
async def test_anthropic_llm_create(model_name):
    model = AsyncAnthropicLLMModel(model_name=model_name, temperature=0)
    result, api_status = await model.create([
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ], max_tokens=4000)
    assert result.strip() == "4"
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "anthropic"

@pytest.mark.asyncio
async def test_anthropic_invalid_model():
    client = AsyncAnthropicLLMModel(model_name="invalid-model")
    result, api_status = await client.create([{"role": "user", "content": "test"}])
    assert result is None
    assert isinstance(api_status, APIStatus)
    assert not api_status.success
    assert api_status.error_info is not None
    assert "not_found_error" in api_status.error_info.error_message
    assert api_status.component == "anthropic"

@pytest.mark.asyncio
async def test_anthropic_llm_create_with_response_model():
    model = AsyncAnthropicLLMModel(
        model_name="claude-3-5-sonnet-20241022", 
        temperature=0,
        response_model=SimpleResponse
    )
    result, api_status = await model.create([
        {"role": "user", "content": "Return the number 4 as a JSON object with the key 'answer'. Respond with only valid JSON."}
    ])
    
    assert isinstance(result, SimpleResponse)
    assert result.answer == 4
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "anthropic"

# LLMModel wrapper tests
def test_llm_model_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMModel(provider="invalid")

@pytest.mark.asyncio
async def test_llm_model_invalid_openai_model():
    model = LLMModel(provider="openai", model_name="invalid-model")
    response, api_status = await model.create([{"role": "user", "content": "test"}])
    assert response is None
    assert isinstance(api_status, APIStatus)
    assert not api_status.success
    assert api_status.error_info is not None
    assert "model_not_found" in api_status.error_info.error_message
    assert api_status.component == "openai"
    assert api_status.error_info.error_type is not None
    assert api_status.error_info.stacktrace is not None
    assert api_status.error_info.file_path is not None

@pytest.mark.asyncio
async def test_llm_model_invalid_anthropic_model():
    model = LLMModel(provider="anthropic", model_name="invalid-model")
    response, api_status = await model.create([{"role": "user", "content": "test"}])
    assert response is None
    assert isinstance(api_status, APIStatus)
    assert not api_status.success
    assert api_status.error_info is not None
    assert "not_found_error" in api_status.error_info.error_message
    assert api_status.component == "anthropic"
    assert api_status.error_info.error_type is not None
    assert api_status.error_info.stacktrace is not None
    assert api_status.error_info.file_path is not None

@pytest.mark.asyncio
async def test_successful_call_error_info():
    model = LLMModel(provider="openai", model_name="gpt-4-1106-preview")
    result, api_status = await model.create([{"role": "user", "content": "Say 'test'"}])
    assert result is not None
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "openai"

@pytest.mark.parametrize("provider,model_name", [
    ("openai", "gpt-4-1106-preview"),
    ("openai", "o1-2024-12-17"),
    ("anthropic", "claude-3-5-sonnet-20241022"),
])
def test_llm_model_valid_models(provider, model_name):
    model = LLMModel(provider=provider, model_name=model_name)
    assert model.model.model_name == model_name

@pytest.mark.asyncio
async def test_llm_model_create_with_response_model():
    model = LLMModel(
        provider="openai",
        model_name="gpt-4o-2024-08-06",
        response_model=SimpleResponse
    )
    response, api_status = await model.create([
        {"role": "user", "content": "Return the number 4 as a JSON object with the key 'answer'. Respond with only valid JSON."}
    ])
    assert isinstance(response, SimpleResponse)
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert response.answer == 4

@pytest.mark.asyncio
async def test_parallel_api_calls():
    model = AsyncOpenAILLMModel(model_name="gpt-4-1106-preview", temperature=0, n_parallel_api_calls=3)
    tasks = []
    for i in range(5):  # Test with 5 parallel calls
        tasks.append(model.create([
            {"role": "user", "content": f"What is {i}+{i}? Answer with just the number."}
        ]))
    
    responses = await asyncio.gather(*tasks)
    for i, (result, api_status) in enumerate(responses):
        assert result.strip() == str(i + i)
        assert isinstance(api_status, APIStatus)
        assert api_status.success
        assert api_status.error_info is None
        assert api_status.component == "openai" 