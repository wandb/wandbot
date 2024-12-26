"""Chat model implementation using LiteLLM."""
from typing import List, Dict, Any, Optional
import litellm

class ChatModel:
    """Chat model using LiteLLM for provider-agnostic interface."""
    
    VALID_ROLES = {"system", "user", "assistant"}

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        fallback_models: Optional[List[str]] = None,
        num_retries: int = 3,
        timeout: int = 30,
        max_backoff: int = 60,
    ):
        """Initialize chat model.
        
        Args:
            model_name: Name of the model to use, in format "provider/model"
                e.g., "openai/gpt-4", "anthropic/claude-3", "gemini/gemini-pro"
            temperature: Sampling temperature between 0 and 1
            fallback_models: List of model names to try if primary model fails
            num_retries: Number of times to retry on retryable errors
            timeout: Request timeout in seconds
            max_backoff: Maximum backoff time between retries in seconds
        """
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")

        self.model_name = model_name
        self.temperature = temperature
        self.fallback_models = fallback_models or []
        self.num_retries = num_retries
        self.timeout = timeout
        self.max_backoff = max_backoff

        # Configure LiteLLM
        litellm.drop_params = True  # Remove unsupported params
        litellm.set_verbose = False
        litellm.success_callback = []
        litellm.failure_callback = []
        
        # Configure fallbacks
        litellm.model_fallbacks = {}  # Reset fallbacks
        litellm.fallbacks = False  # Reset fallbacks flag
        if self.fallback_models:
            litellm.model_fallbacks = {
                self.model_name: self.fallback_models
            }
            litellm.fallbacks = True

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert messages to provider-specific format."""
        if not messages:
            raise ValueError("No messages provided")

        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(f"Invalid message format: {msg}")
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message missing role or content: {msg}")
            if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
                raise ValueError(f"Role and content must be strings: {msg}")
            if msg["role"] not in self.VALID_ROLES:
                raise ValueError(f"Invalid role: {msg['role']}")

        # LiteLLM handles provider-specific message formats
        return messages

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Generate a response from the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing:
                - content: The generated response text
                - total_tokens: Total tokens used
                - prompt_tokens: Tokens used in the prompt
                - completion_tokens: Tokens used in the completion
                - error: Error information if request failed
                - model_used: Name of the model that generated the response
        """
        # Handle empty messages
        if not messages:
            return {
                "content": "",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": {
                    "type": "ValueError",
                    "message": "No messages provided",
                    "retryable": False
                },
                "model_used": self.model_name
            }

        try:
            # Convert messages to provider-specific format
            converted_messages = self._convert_messages(messages)

            # Use LiteLLM's built-in fallback mechanism
            response = litellm.completion(
                model=self.model_name,
                messages=converted_messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                num_retries=self.num_retries,
                fallbacks=self.fallback_models  # LiteLLM's fallback mechanism
            )

            return {
                "content": response.choices[0].message.content,
                "error": None,
                "model_used": response.model
            }

        except Exception as e:
            return {
                "content": "",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e)
                },
                "model_used": self.model_name
            }