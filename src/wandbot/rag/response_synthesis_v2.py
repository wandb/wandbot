from typing import Any, Dict, List, Optional
import weave
from wandbot.rag.utils import combine_documents, create_query_str
from wandbot.schema.retrieval import RetrievalResult
from wandbot.models.llm import LLMModel
from wandbot.schema.api_status import APIStatus
from wandbot.rag.response_synthesis import RESPONSE_SYNTHESIS_SYSTEM_PROMPT, RESPONSE_SYNTHESIS_PROMPT_MESSAGES
from langchain_core.prompts import ChatPromptTemplate

class ResponseSynthesizerV2:
    def __init__(
        self,
        model: str,
        temperature: float,
        fallback_model: str,
        fallback_temperature: float,
    ):
        self.model = LLMModel("openai", model_name=model, temperature=temperature)
        self.fallback_model = LLMModel("openai", model_name=fallback_model, temperature=fallback_temperature)
        self.prompt = ChatPromptTemplate.from_messages(RESPONSE_SYNTHESIS_PROMPT_MESSAGES)
        self._last_formatted_messages = None
        self._last_formatted_input = None

    def _format_input(self, inputs: RetrievalResult) -> Dict[str, str]:
        """Format the input data for the prompt template."""
        return {
            "query_str": create_query_str({
                "standalone_query": inputs.retrieval_info["query"],
                "language": inputs.retrieval_info["language"],
                "intents": inputs.retrieval_info["intents"],
                "sub_queries": inputs.retrieval_info["sub_queries"]
            }),
            "context_str": combine_documents(inputs.documents),
        }

    def get_formatted_messages(self, inputs: RetrievalResult) -> List[Dict[str, str]]:
        """Get the formatted messages that would be sent to OpenAI."""
        # Format the input data if not already formatted
        if self._last_formatted_input is None:
            self._last_formatted_input = self._format_input(inputs)
        
        # Get the formatted messages from the prompt template
        messages = self.prompt.format_messages(**self._last_formatted_input)
        
        # Convert to dict format that OpenAI expects
        formatted_messages = [{"role": msg.type, "content": msg.content} for msg in messages]
        self._last_formatted_messages = formatted_messages
        return formatted_messages

    @weave.op
    async def __call__(self, inputs: RetrievalResult) -> Dict[str, Any]:
        # Reset the cached formatted input
        self._last_formatted_input = None
        
        # Format messages
        messages = self.get_formatted_messages(inputs)
        
        # Try primary model first
        response, api_status = await self.model.create(messages=messages)
        
        # If primary model fails, try fallback
        if not api_status.success:
            response, api_status = await self.fallback_model.create(messages=messages)
        
        # If both models fail, return error
        if not api_status.success:
            return {
                "query_str": self._last_formatted_input["query_str"],
                "context_str": self._last_formatted_input["context_str"],
                "response": None,
                "response_model": None,
                "response_synthesis_llm_messages": self._last_formatted_messages,
                "api_status": api_status
            }
        
        # Return successful response
        return {
            "query_str": self._last_formatted_input["query_str"],
            "context_str": self._last_formatted_input["context_str"],
            "response": response,
            "response_model": api_status.component,
            "response_synthesis_llm_messages": self._last_formatted_messages,
            "api_status": api_status
        } 