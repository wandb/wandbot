import pytest
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
import json

from wandbot.models.llm import (
    AsyncOpenAILLMModel,
    AsyncAnthropicLLMModel,
    LLMModel,
    extract_system_and_messages
)

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
    response = await model.create([
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ])
    assert response.strip() == "4"

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", openai_models)
async def test_openai_llm_create_with_response_model(model_name):
    model = AsyncOpenAILLMModel(model_name=model_name, temperature=0)
    response = await model.create([
        {"role": "user", "content": "Return the number 4 as a JSON object with the key 'answer'. Respond with only valid JSON."}
    ], response_model=SimpleResponse)
    
    assert isinstance(response, SimpleResponse)
    assert response.answer == 4

@pytest.mark.asyncio
async def test_openai_invalid_model():
    model = AsyncOpenAILLMModel(model_name="invalid-model", temperature=0)
    with pytest.raises(Exception):  # OpenAI will raise an error on API call
        await model.create([{"role": "user", "content": "test"}])

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
    response = await model.create([
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ], max_tokens=4000)
    assert response.strip() == "4"

@pytest.mark.asyncio
async def test_anthropic_invalid_model():
    model = AsyncAnthropicLLMModel(model_name="invalid-model", temperature=0)
    with pytest.raises(Exception):  # Anthropic will raise an error on API call
        await model.create([{"role": "user", "content": "test"}], max_tokens=4000)

@pytest.mark.asyncio
async def test_anthropic_llm_create_with_response_model():
    model = AsyncAnthropicLLMModel(model_name="claude-3-5-sonnet-20241022", temperature=0)
    response = await model.create([
        {"role": "user", "content": "Return the number 4 as a JSON object with the key 'answer'. Respond with only valid JSON."}
    ], response_model=SimpleResponse)
    
    assert isinstance(response, SimpleResponse)
    assert response.answer == 4

# LLMModel wrapper tests
def test_llm_model_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMModel(provider="invalid")

@pytest.mark.asyncio
async def test_llm_model_invalid_openai_model():
    model = LLMModel(provider="openai", model_name="invalid-model")
    with pytest.raises(Exception):  # Will raise on API call
        await model.create([{"role": "user", "content": "test"}])

@pytest.mark.asyncio
async def test_llm_model_invalid_anthropic_model():
    model = LLMModel(provider="anthropic", model_name="invalid-model")
    with pytest.raises(Exception):  # Will raise on API call
        await model.create([{"role": "user", "content": "test"}])

@pytest.mark.parametrize("provider,model_name", [
    ("openai", "gpt-4-1106-preview"),
    ("openai", "o1-2024-12-17"),
    ("anthropic", "claude-3-5-sonnet-20241022"),
])
def test_llm_model_valid_models(provider, model_name):
    model = LLMModel(provider=provider, model_name=model_name)
    assert model.model.model_name == model_name

@pytest.mark.asyncio
async def test_llm_model_call():
    model = LLMModel(provider="openai", model_name="gpt-4-1106-preview")
    response = await model([
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ])
    assert response.strip() == "4"

@pytest.mark.asyncio
async def test_llm_model_call_with_response_model():
    model = LLMModel(provider="openai", model_name="gpt-4o-2024-08-06")
    response = await model([
        {"role": "user", "content": "Return the number 4 as a JSON object with the key 'answer'. Respond with only valid JSON."}
    ], response_model=SimpleResponse)
    assert isinstance(response, SimpleResponse)
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
    for i, response in enumerate(responses):
        assert response.strip() == str(i + i) 