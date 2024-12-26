from typing import List, Dict, Any, Optional
from openai import OpenAI, OpenAIError, APIError, RateLimitError, APIConnectionError

from .base import ChatModel, ModelError

class OpenAIChatModel(ChatModel):
    """OpenAI chat model implementation."""
    
    def _map_error(self, error: Exception) -> ModelError:
        """Map OpenAI errors to standardized ModelError."""
        if isinstance(error, OpenAIError):
            return ModelError(
                type="api_error",
                message=str(error),
                retryable=True
            )
        elif isinstance(error, ValueError):
            return ModelError(
                type="invalid_input",
                message=str(error),
                retryable=False
            )
        else:
            return ModelError(
                type="unknown_error",
                message=str(error),
                retryable=True
            )

    def __init__(
        self,
        model_name: str = "gpt-4-0125-preview",
        temperature: float = 0.1,
        fallback_model: Optional['ChatModel'] = None,
    ):
        super().__init__(model_name, temperature, fallback_model)
        self.client = OpenAI()

    def _generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        # Validate input
        if not messages:
            return self._create_error_response(ModelError(
                type="invalid_input",
                message="No messages provided",
                retryable=False
            ))

        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return self._create_error_response(ModelError(
                    type="invalid_input",
                    message="Invalid message format",
                    retryable=False
                ))
            if msg["role"] not in ["system", "user", "assistant"]:
                return self._create_error_response(ModelError(
                    type="invalid_input",
                    message=f"Invalid role: {msg['role']}",
                    retryable=False
                ))

        # Convert messages to OpenAI format, using "developer" instead of "system"
        openai_messages = []
        for msg in messages:
            role = "developer" if msg["role"] == "system" else msg["role"]
            openai_messages.append({
                "role": role,
                "content": msg["content"]
            })

        # Create completion with OpenAI's API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        return {
            "content": response.choices[0].message.content,
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "error": None
        }

    @property
    def system_role_key(self) -> str:
        """Return the key used for system role in messages."""
        return "developer"  # OpenAI now uses "developer" instead of "system"