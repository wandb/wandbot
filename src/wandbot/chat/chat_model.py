"""Chat model descriptor for LiteLLM."""
import litellm

class ChatModel:
    """Chat model descriptor that wraps LiteLLM for provider-agnostic interface."""
    
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, obj_type=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        """Configure LiteLLM with the given model settings.
        
        Args:
            value: Dictionary containing:
                - model_name: Name of the model to use (e.g., "openai/gpt-4")
                - temperature: Sampling temperature between 0 and 1
                - fallback_models: Optional list of fallback models
        """
        if not 0 <= value["temperature"] <= 1:
            raise ValueError("Temperature must be between 0 and 1")

        # Configure LiteLLM
        litellm.drop_params = True  # Remove unsupported params
        litellm.set_verbose = False
        litellm.success_callback = []
        litellm.failure_callback = []
        
        # Configure fallbacks
        litellm.model_fallbacks = {}  # Reset fallbacks
        litellm.fallbacks = False  # Reset fallbacks flag
        if value.get("fallback_models"):
            litellm.model_fallbacks = {
                value["model_name"]: value["fallback_models"]
            }
            litellm.fallbacks = True

        # Create completion function that matches LangChain's ChatOpenAI interface
        def completion_fn(messages, **kwargs):
            try:
                response = litellm.completion(
                    model=value["model_name"],
                    messages=messages,
                    temperature=value["temperature"],
                    num_retries=self.max_retries,
                    **kwargs
                )
                return response
            except Exception as e:
                # Return error response that matches LangChain's interface
                return type("Response", (), {
                    "choices": [
                        type("Choice", (), {
                            "message": type("Message", (), {
                                "content": ""
                            })()
                        })()
                    ],
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e)
                    },
                    "model": value["model_name"]
                })()

        setattr(obj, self.private_name, completion_fn)