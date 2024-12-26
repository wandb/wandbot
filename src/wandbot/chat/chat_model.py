"""Chat model implementation using LiteLLM."""
from typing import List, Dict, Any
import litellm

class ChatModel:
    """Chat model using LiteLLM for provider-agnostic interface."""
    
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
        """
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
        # Convert messages for OpenAI-style models
        if "openai" in self.model_name:
            messages = [
                {
                    "role": "developer" if msg["role"] == "system" else msg["role"],
                    "content": msg["content"]
                }
                for msg in messages
            ]

        try:
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