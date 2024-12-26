from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass

@dataclass
class ModelError:
    """Structured error information."""
    type: str  # e.g., "auth_error", "rate_limit", "context_length", etc.
    message: str  # Human-readable error message
    code: Optional[str] = None  # Provider-specific error code if available
    retryable: bool = False  # Whether the error is potentially retryable

class ChatModel(ABC):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        fallback_model: Optional['ChatModel'] = None,
    ):
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        
        self.model_name = model_name
        self.temperature = temperature
        self.fallback_model = fallback_model

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Generate a response from the model with fallback support.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing:
                - content: The generated response text (empty string if error)
                - total_tokens: Total tokens used (0 if error)
                - prompt_tokens: Tokens used in the prompt (0 if error)
                - completion_tokens: Tokens used in the completion (0 if error)
                - error: None if successful, ModelError instance if failed
                - model_used: Name of the model that generated the response
        """
        try:
            response = self._generate_response(messages, max_tokens)
            response["model_used"] = self.model_name
            return response
        except Exception as e:
            error = self._map_error(e) if hasattr(self, '_map_error') else ModelError(
                type="unknown_error",
                message=str(e),
                retryable=True
            )
            
            # If error is retryable and we have a fallback model, try it
            if error.retryable and self.fallback_model:
                try:
                    fallback_response = self.fallback_model.generate_response(messages, max_tokens)
                    if not fallback_response.get("error"):
                        return fallback_response
                except Exception:
                    # If fallback fails, return original error
                    pass
            
            return self._create_error_response(error)

    @abstractmethod
    def _generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Internal method to generate a response from the model.
        
        This method should be implemented by each model provider.
        """
        pass

    @property
    @abstractmethod
    def system_role_key(self) -> str:
        """Return the key used for system role in messages."""
        pass

    def _create_error_response(self, error: ModelError) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "content": "",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": error,
            "model_used": self.model_name
        }