from typing import List, Dict, Any, Optional
from wandbot.chat.models.base import ChatModel, ModelError

class MockOpenAIModel(ChatModel):
    """Mock OpenAI model for testing."""
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        fallback_model: Optional['ChatModel'] = None,
    ):
        super().__init__(model_name, temperature, fallback_model)

    def _generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        # This is a mock that just returns a fixed response
        return {
            "content": "Response from mock OpenAI",
            "total_tokens": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "error": None
        }

    @property
    def system_role_key(self) -> str:
        return "system"