from typing import List, Dict, Any, Optional
import anthropic
from anthropic._types import NOT_GIVEN

from .base import ChatModel, ModelError

class AnthropicChatModel(ChatModel):
    """Anthropic Claude model implementation."""
    
    ERROR_MAPPING = {
        anthropic.APIError: ("api_error", "Anthropic API error", True),
        anthropic.APIConnectionError: ("connection_error", "Connection to Anthropic failed", True),
        anthropic.APIResponseValidationError: ("validation_error", "Invalid API response", False),
        anthropic.APIStatusError: ("status_error", "API status error", True),
        anthropic.AuthenticationError: ("auth_error", "Invalid API key", False),
        anthropic.BadRequestError: ("invalid_request", "Invalid request parameters", False),
        anthropic.InternalServerError: ("server_error", "Anthropic server error", True),
        anthropic.NotFoundError: ("not_found", "Resource not found", False),
        anthropic.PermissionDeniedError: ("permission_denied", "Permission denied", False),
        anthropic.RateLimitError: ("rate_limit", "Rate limit exceeded", True),
        ValueError: ("invalid_input", "Invalid input parameters", False),
    }

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        temperature: float = 0.1,
        fallback_model: Optional['ChatModel'] = None,
    ):
        super().__init__(model_name, temperature, fallback_model)
        self.client = anthropic.Anthropic()

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

        # Extract system message if present
        system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        
        # Convert remaining messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue  # Handle separately
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            elif role == "assistant":
                anthropic_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg["content"]}]
                })

        # Create message with Anthropic's API
        response = self.client.messages.create(
            model=self.model_name,
            messages=anthropic_messages,
            system=system_msg if system_msg is not None else NOT_GIVEN,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        # Extract content from response
        content = response.content[0].text if response.content else ""

        return {
            "content": content,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "error": None
        }

    @property
    def system_role_key(self) -> str:
        """Return the key used for system role in messages."""
        return "system"  # For compatibility with message format, though handled separately