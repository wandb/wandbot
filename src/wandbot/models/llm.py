import asyncio
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Union

import weave
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from pydantic import BaseModel

from wandbot.schema.api_status import APIStatus
from wandbot.utils import ErrorInfo, get_error_file_path, get_logger

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

def extract_google_system_and_messages(messages: List[Dict[str, Any]]) -> tuple:
    """Extract system message and convert remaining messages to Google GenAI Content format."""
    system_msg = None
    chat_messages: List[Dict[str, Any]] = []  # Change type hint for clarity

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if not role or not content:
            logger.warning(f"Skipping message with missing role or content: {msg}")
            continue

        if role == "system" or role == "developer":
            if system_msg is None:  # Take first system message only
                system_msg = content
            else:
                 logger.warning("Multiple system/developer messages found. Only the first one will be used as system instruction.")
        elif role == "assistant":  # Google uses 'model' role
            chat_messages.append({"role": "model", "parts": [{"text": content}]})
        elif role == "user":
            chat_messages.append({"role": "user", "parts": [{"text": content}]})
        else:
            logger.warning(f"Unsupported role encountered: {role}. Skipping message.")

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
            # Pre-process messages: Convert "developer" role back to "system" if found
            # This handles potential in-place modification from other providers (like OpenAI beta)
            processed_messages = []
            for msg in messages:
                if msg.get("role") == "developer":
                    processed_messages.append({"role": "system", "content": msg.get("content")})
                    logger.debug("Converted 'developer' role to 'system' for Anthropic call.")
                else:
                    processed_messages.append(msg)
            
            # Use the processed messages list for extraction
            system_msg, chat_messages = extract_system_and_messages(processed_messages)
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

class AsyncGoogleGenAILLMModel(BaseLLMModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
        self.thinking_budget = kwargs.get("thinking_budget")

    @weave.op
    async def create(self, 
                    messages: List[Dict[str, Any]],
                    max_tokens: int = 4000) -> tuple[Union[str, BaseModel], APIStatus]:
        api_status = APIStatus(component="gemini", success=True)
        try:
            system_instruction_content, chat_messages = extract_google_system_and_messages(messages)
            
            # Prepare arguments for GenerateContentConfig
            generation_config_args = {
                "temperature": self.temperature if self.temperature > 0 else 1.0,
                "max_output_tokens": max_tokens
            }

            generation_config_args["system_instruction"] = system_instruction_content if system_instruction_content else None
                        
            if self.response_model:
                generation_config_args["response_mime_type"] = "application/json"
                generation_config_args["response_schema"] = self.response_model
            

            if "2.5" in self.model_name and self.thinking_budget is not None:
                generation_config_args["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=self.thinking_budget)    

            # Create the config object
            gen_config = genai_types.GenerateContentConfig(**generation_config_args)

            async with self.semaphore: # Apply semaphore for rate limiting
                # Use client.aio.models.generate_content for the async call
                response = await self.client.aio.models.generate_content(
                    model=self.model_name, 
                    contents=chat_messages, 
                    config=gen_config,  # Pass config object here
                )

            
            if self.response_model:
                 # Gemini API with JSON mode often returns the JSON directly in the text part
                 # The 'parsed' attribute might not be standard or guaranteed across versions/models
                try:
                    # Attempt to parse the text content as JSON
                    json_content = json.loads(response.text)
                    return self.response_model.model_validate(json_content), api_status
                except json.JSONDecodeError:
                    # If parsing fails, try cleaning potential markdown and re-parsing
                    logger.warning("Google GenAI response was not valid JSON initially. Attempting to clean and re-parse.")
                    cleaned_content = clean_json_string(response.text)
                    try:
                        json_content = json.loads(cleaned_content)
                        return self.response_model.model_validate(json_content), api_status
                    except Exception as json_e:
                         logger.error(f"Failed to parse Google GenAI JSON response even after cleaning: {json_e}")
                         raise json_e # Re-raise the parsing error after logging
                except Exception as validation_e:
                    logger.error(f"Failed to validate Google GenAI JSON response against schema: {validation_e}")
                    raise validation_e # Re-raise the validation error
            else:
                 # For standard text responses
                 return response.text, api_status
        
        except Exception as e:
            error_info = ErrorInfo(
                component="gemini",
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2])
            )
            return None, APIStatus(
                component="gemini",
                success=False,
                error_info=error_info
            )


class LLMModel:
    PROVIDER_MAP = {
        "openai": AsyncOpenAILLMModel,
        "anthropic": AsyncAnthropicLLMModel,
        "google": AsyncGoogleGenAILLMModel,
    }

    def __init__(self, provider: str, **kwargs):
        provider = provider.lower()
        if provider not in self.PROVIDER_MAP:
            raise ValueError(f"Unsupported provider: {provider}. Choose from {list(self.PROVIDER_MAP.keys())}")
        
        try:
            self.model = self.PROVIDER_MAP[provider](**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to initialize {provider} model: {str(e)}")

    @property
    def model_name(self) -> str:
        return self.model.model_name

    async def create(self, 
                    messages: List[Dict[str, Any]], 
                    **kwargs) -> tuple[Union[str, BaseModel], APIStatus]:
        try:
            async with self.model.semaphore: # Use the specific model's semaphore
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
