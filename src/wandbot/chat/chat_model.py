"""Chat model implementation using LiteLLM."""
from typing import List, Dict, Any
import litellm

class ChatModel:
    """Chat model using LiteLLM for provider-agnostic interface."""
    
    VALID_ROLES = {"system", "user", "assistant"}

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
    ):
        """Initialize chat model.
        
        Args:
            model_name: Name of the model to use, in format "provider/model"
                e.g., "openai/gpt-4", "anthropic/claude-3", "gemini/gemini-pro"
            temperature: Sampling temperature between 0 and 1
            
        Raises:
            ValueError: If temperature is not between 0 and 1
        """
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")

        self.model_name = model_name
        self.temperature = temperature

        # Configure LiteLLM
        litellm.drop_params = True  # Remove unsupported params
        litellm.set_verbose = False

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
        # Validate messages
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

            # Handle provider-specific message formats
            if "openai" in self.model_name:
                # OpenAI: Convert system to developer role
                messages = [
                    {
                        "role": "developer" if msg["role"] == "system" else msg["role"],
                        "content": msg["content"]
                    }
                    for msg in messages
                ]
            elif "anthropic" in self.model_name:
                # Anthropic: Handle system message separately
                system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                messages = [msg for msg in messages if msg["role"] != "system"]
                if system_msg:
                    messages.insert(0, {"role": "system", "content": system_msg})
            elif "gemini" in self.model_name:
                # Gemini: Prepend system message to first user message
                system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                if system_msg:
                    messages = [msg for msg in messages if msg["role"] != "system"]
                    for msg in messages:
                        if msg["role"] == "user":
                            msg["content"] = f"{system_msg}\n\n{msg['content']}"
                            break

            # Generate response
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )

            return {
                "content": response.choices[0].message.content,
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "error": None,
                "model_used": response.model
            }

        except Exception as e:
            # Determine if error is retryable
            error_msg = str(e).lower()
            retryable = any(
                err_type in error_msg
                for err_type in ["timeout", "rate limit", "server", "connection"]
            )

            return {
                "content": "",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "retryable": retryable
                },
                "model_used": self.model_name
            }