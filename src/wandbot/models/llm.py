import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
import weave
import traceback
import sys

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from wandbot.utils import get_logger, get_error_file_path, ErrorInfo
from wandbot.schema.api_status import APIStatus

logger = get_logger(__name__)

class LLMError(BaseModel):
    error: bool
    error_message: str

def extract_system_and_messages(messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Extract system message and convert remaining messages to Anthropic format."""
    system_msg = None
    chat_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            if system_msg is None:  # Take first system message only
                system_msg = msg["content"]
        else:
            chat_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    return system_msg, chat_messages


def add_json_response_model_to_messages(response_model: BaseModel) -> List[Dict[str, Any]]:
    response_schema = json.dumps(response_model.model_json_schema(), indent=2)
    return [{"role": "user",
            "content": "You must respond with a valid JSON object that matches this schema.\
Do not include any explanations or text outside the JSON object:\n\n" + response_schema}]


def clean_json_string(json_str: str) -> str:
    """Clean a JSON string that might be wrapped in markdown code blocks.
    
    Args:
        json_str: Raw string that may contain a JSON object wrapped in code blocks
        
    Returns:
        Cleaned JSON string with code blocks and language identifiers removed
    """
    json_str = json_str.strip()
    if json_str.startswith("```"):
        json_str = json_str.split("\n", 1)[1].rsplit("\n", 1)[0]
    if json_str.startswith("json"):
        json_str = json_str.split("\n", 1)[1]
    return json_str.strip()


class BaseLLMModel:
    def __init__(self, 
                 model_name: str,
                 temperature: float = 0,
                 response_model: Optional[BaseModel] = None,
                 n_parallel_api_calls: int = 50,
                 max_retries: int = 3,
                 timeout: int = 30,
                 **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.response_model = response_model
        self.n_parallel_api_calls = n_parallel_api_calls
        self.max_retries = max_retries
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(n_parallel_api_calls)

    async def create(self, 
                    messages: List[Dict[str, Any]], 
                    **kwargs) -> tuple[Union[str, BaseModel], APIStatus]:
        raise NotImplementedError("Subclasses must implement create method")

class AsyncOpenAILLMModel(BaseLLMModel):
    JSON_MODELS = [
        "gpt-4-",  # All gpt-4- models
        "o1-mini"  # o1-mini models
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            max_retries=self.max_retries,
            timeout=self.timeout
        )

    @weave.op
    async def create(self, 
                    messages: List[Dict[str, Any]]) -> tuple[Union[str, BaseModel], APIStatus]:
        api_status = APIStatus(component="openai", success=True)
        try:
            api_params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": messages,
            }
            if api_params["temperature"] == 0:
                api_params["temperature"] = 0.1

            if self.model_name.startswith("o"):
                api_params.pop("temperature", None)
            
            if self.response_model:
                # For models that don't support the Structure Outputs api:
                if any(self.model_name.startswith(prefix) for prefix in self.JSON_MODELS):
                    if not self.model_name.startswith("o1-mini"):  # o1-mini doesn't support response_format either
                        api_params["response_format"] = {"type": "json_object"}
                    
                    api_params["messages"] += add_json_response_model_to_messages(self.response_model)
                    response = await self.client.chat.completions.create(**api_params)
                    json_str = response.choices[0].message.content
                    json_str = clean_json_string(json_str)
                    return self.response_model.model_validate_json(json_str), api_status
                # Else use the Structure Outputs api
                else:
                    api_params["response_format"] = self.response_model
                    for msg in api_params["messages"]:
                        if msg["role"] == "system":
                            msg["role"] = "developer"
                    
                    response = await self.client.beta.chat.completions.parse(**api_params)
                    return response.choices[0].message.parsed, api_status
            else:
                response = await self.client.chat.completions.create(**api_params)
                return response.choices[0].message.content, api_status
        except Exception as e:
            error_info = ErrorInfo(
                component="openai",
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2])
            )
            return None, APIStatus(
                component="openai",
                success=False,
                error_info=error_info
            )

class AsyncAnthropicLLMModel(BaseLLMModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_retries=self.max_retries,
            timeout=self.timeout
        )

    @weave.op
    async def create(self, 
                    messages: List[Dict[str, Any]], 
                    max_tokens: int = 4000) -> tuple[Union[str, BaseModel], APIStatus]:
        api_status = APIStatus(component="anthropic", success=True)
        try:
            system_msg, chat_messages = extract_system_and_messages(messages)
            api_params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": chat_messages,
                "max_tokens": max_tokens
            }
            if api_params["temperature"] == 0:
                api_params["temperature"] = 0.1

            if system_msg:
                api_params["system"] = system_msg

            if self.response_model:
                api_params["messages"] += add_json_response_model_to_messages(self.response_model)

            response = await self.client.messages.create(**api_params)
            content = response.content[0].text

            if self.response_model:
                json_str = clean_json_string(content)
                return self.response_model.model_validate_json(json_str), api_status
            return content, api_status
        except Exception as e:
            error_info = ErrorInfo(
                component="anthropic",
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2])
            )
            return None, APIStatus(
                component="anthropic",
                success=False,
                error_info=error_info
            )


class LLMModel:
    PROVIDER_MAP = {
        "openai": AsyncOpenAILLMModel,
        "anthropic": AsyncAnthropicLLMModel
    }

    def __init__(self, provider: str, **kwargs):
        provider = provider.lower()
        if provider not in self.PROVIDER_MAP:
            raise ValueError(f"Unsupported provider: {provider}. Choose from {list(self.PROVIDER_MAP.keys())}")
        
        try:
            self.model = self.PROVIDER_MAP[provider](**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to initialize {provider} model: {str(e)}")

    async def create(self, 
                    messages: List[Dict[str, Any]], 
                    **kwargs) -> tuple[Union[str, BaseModel], APIStatus]:
        try:
            response, api_status = await self.model.create(
                messages=messages,
                **kwargs
            )
            return response, api_status
        except Exception as e:
            logger.error(f"LLMModel: Error in LLM API call: {str(e)}")
            error_info = ErrorInfo(
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2]),
                component="llm"
            )
            return None, APIStatus(
                component="llm",
                success=False,
                error_info=error_info
            )
