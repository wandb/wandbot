from typing import List, Dict, Any

from anthropic import Anthropic

from .base import ChatModel

class AnthropicChatModel(ChatModel):
    def __init__(self, model_name: str = "claude-3-opus-20240229", temperature: float = 0.1):
        super().__init__(model_name, temperature)
        self.client = Anthropic()

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})

        response = self.client.messages.create(
            model=self.model_name,
            messages=anthropic_messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        return {
            "content": response.content[0].text,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    @property
    def system_role_key(self) -> str:
        return "system"  # Will be converted to assistant role in generate_response