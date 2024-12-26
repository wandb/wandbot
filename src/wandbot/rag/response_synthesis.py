from typing import Dict, List, Any

from wandbot.chat.models import OpenAIChatModel, GeminiChatModel, AnthropicChatModel
from wandbot.utils import get_logger

logger = get_logger(__name__)

class ResponseSynthesizer:
    def __init__(
        self,
        model: str = "openai/gpt-4-0125-preview",
        temperature: float = 0.1,
        fallback_model: str = "openai/gpt-4-0125-preview",
        fallback_temperature: float = 0.1,
    ):
        self.model_str = model
        self.temperature = temperature
        self.fallback_model_str = fallback_model
        self.fallback_temperature = fallback_temperature
        
        # Initialize primary model
        self.model = self._create_model(model, temperature)
        # Initialize fallback model
        self.fallback_model = self._create_model(fallback_model, fallback_temperature)

    def _create_model(self, model_str: str, temperature: float):
        provider, model_name = model_str.split("/", 1)
        
        if provider == "openai":
            return OpenAIChatModel(model_name=model_name, temperature=temperature)
        elif provider == "gemini":
            return GeminiChatModel(model_name=model_name, temperature=temperature)
        elif provider == "anthropic":
            return AnthropicChatModel(model_name=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _create_messages(
        self, context_str: str, query: str, chat_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        messages = []
        
        # System message with context
        system_prompt = (
            "You are WandBot, a helpful AI assistant for Weights & Biases (W&B). "
            "Answer the question based on the context below. If you don't know the answer, "
            "say that you don't know. Use the following format:\n\n"
            "Context: relevant context from W&B documentation\n\n"
            "Question: the user's question\n\n"
            "Answer: your response\n\n"
            "Here's the context:\n\n"
            f"{context_str}\n\n"
            "Remember:\n"
            "1. Only answer based on the context provided\n"
            "2. If the context doesn't contain relevant information, say so\n"
            "3. Keep responses clear and concise\n"
            "4. Include code examples when relevant\n"
            "5. Use markdown formatting for better readability"
        )
        messages.append({"role": self.model.system_role_key, "content": system_prompt})

        # Add chat history
        for msg in chat_history:
            messages.append({"role": "user", "content": msg[0]})
            messages.append({"role": "assistant", "content": msg[1]})

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    def __call__(self, retrieval_results: Dict[str, Any]) -> Dict[str, Any]:
        context = retrieval_results["context"]
        query = retrieval_results["query"]
        chat_history = retrieval_results.get("chat_history", [])

        # Format context string
        context_str = "\n\n".join(
            f"Source: {doc.metadata['source']}\n{doc.page_content}"
            for doc in context
        )

        messages = self._create_messages(context_str, query, chat_history)

        try:
            # Try primary model first
            response = self.model.generate_response(messages)
            model_used = self.model_str
        except Exception as e:
            logger.warning(f"Primary model failed: {e}. Falling back to backup model.")
            try:
                # Fall back to backup model
                response = self.fallback_model.generate_response(messages)
                model_used = self.fallback_model_str
            except Exception as e:
                logger.error(f"Both models failed. Last error: {e}")
                raise

        return {
            "response": response["content"],
            "context_str": context_str,
            "response_prompt": messages[0]["content"],
            "response_model": model_used,
            "total_tokens": response["total_tokens"],
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
        }