from typing import List, Dict, Any

from openai import OpenAI

from .base import ChatModel

class OpenAIChatModel(ChatModel):
    def __init__(self, model_name: str = "gpt-4-0125-preview", temperature: float = 0.1):
        super().__init__(model_name, temperature)
        self.client = OpenAI()

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        
        return {
            "content": response.choices[0].message.content,
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    @property
    def system_role_key(self) -> str:
        return "system"