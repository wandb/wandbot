from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from .base import ChatModel, ModelError

class GeminiChatModel(ChatModel):
    ERROR_MAPPING = {
        # Auth and permissions
        google_exceptions.PermissionDenied: ("auth_error", "Invalid API key or authentication failed", False),
        
        # Rate limits and quotas
        google_exceptions.ResourceExhausted: ("rate_limit", "Rate limit or quota exceeded", True),
        
        # Invalid requests
        google_exceptions.InvalidArgument: ("invalid_request", "Invalid request parameters", False),
        ValueError: ("invalid_input", "Invalid input format or parameters", False),
        
        # Network and connectivity
        ConnectionError: ("network_error", "Failed to connect to API", True),
        TimeoutError: ("timeout", "Request timed out", True),
        
        # Server errors
        google_exceptions.InternalServerError: ("server_error", "Gemini API server error", True),
        
        # Model errors
        google_exceptions.NotFound: ("model_error", "Model not found or unavailable", False),
        google_exceptions.FailedPrecondition: ("model_error", "Model is not ready or unavailable", True),
    }

    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0.1,
        fallback_model: Optional['ChatModel'] = None,
    ):
        super().__init__(model_name, temperature, fallback_model)
        try:
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            # Handle initialization errors (e.g., invalid model name)
            error = self._map_error(e)
            raise RuntimeError(f"Failed to initialize Gemini model: {error.message}")

    def _map_error(self, error: Exception) -> ModelError:
        """Map Gemini API errors to standardized ModelError."""
        for error_type, (type_str, msg, retryable) in self.ERROR_MAPPING.items():
            if isinstance(error, error_type):
                return ModelError(
                    type=type_str,
                    message=str(error) or msg,
                    code=getattr(error, 'code', None),
                    retryable=retryable
                )
        
        # Handle safety-related errors
        if hasattr(error, 'prompt_feedback'):
            return ModelError(
                type="safety_error",
                message="Content filtered due to safety concerns",
                code="SAFETY_BLOCK",
                retryable=False
            )
        
        # Default error handling
        return ModelError(
            type="unknown_error",
            message=str(error) or "An unknown error occurred",
            retryable=False
        )

    def _generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})

        # If there was a system message, prepend it to the first user message
        system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        if system_msg and gemini_messages:
            for msg in gemini_messages:
                if msg["role"] == "user":
                    msg["parts"][0] = f"{system_msg}\n\n{msg['parts'][0]}"
                    break

        if not gemini_messages:
            raise ValueError("No valid messages provided")

        # Get response from model
        chat = self.model.start_chat(history=gemini_messages)
        response = chat.send_message(
            gemini_messages[-1]["parts"][0],
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=max_tokens,
            )
        )

        # Check for safety blocks or other content filtering
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            raise ValueError("Content filtered due to safety concerns")

        # Get token counts from usage_metadata
        usage = response.usage_metadata
        
        return {
            "content": response.text,
            "total_tokens": usage.total_token_count,
            "prompt_tokens": usage.prompt_token_count,
            "completion_tokens": usage.candidates_token_count,
            "error": None
        }

    @property
    def system_role_key(self) -> str:
        """Return the key used for system role in messages."""
        return "system"