"""Handles chat interactions for WandBot.

This module contains the Chat class which is responsible for handling chat interactions. 
It includes methods for initializing the chat, loading the storage context from an artifact, 
loading the chat engine, validating and formatting questions, formatting responses, and getting answers. 
It also contains a function for generating a list of chat messages from a given chat history.

Typical usage example:

  config = ChatConfig()
  chat = Chat(config=config)
  chat_history = []
  while True:
      question = input("You: ")
      if question.lower() == "quit":
          break
      else:
          response = chat(
              ChatRequest(question=question, chat_history=chat_history)
          )
          chat_history.append(
              QuestionAnswer(question=question, answer=response.answer)
          )
          print(f"WandBot: {response.answer}")
          print(f"Time taken: {response.time_taken}")
"""

import json
from typing import Any, Dict, List

from llama_index import ServiceContext
from llama_index.callbacks import (
    CallbackManager,
    TokenCountingHandler,
    WandbCallbackHandler,
)
from llama_index.chat_engine import (
    CondensePlusContextChatEngine,
    ContextChatEngine,
)
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.postprocessor import CohereRerank
from llama_index.llms import ChatMessage
from weave.monitoring import StreamTable

import wandb
from wandbot.chat.config import ChatConfig
from wandbot.chat.prompts import load_chat_prompt
from wandbot.chat.query_handler import QueryHandler, ResolvedQuery
from wandbot.chat.retriever import LanguageFilterPostprocessor, Retriever
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.utils import Timer, get_logger, load_service_context

logger = get_logger(__name__)


class Chat:
    """Class for handling chat interactions.

    Attributes:
        config: An instance of ChatConfig containing configuration settings.
        run: An instance of wandb.Run for logging experiment information.
        wandb_callback: An instance of WandbCallbackHandler for handling Wandb callbacks.
        token_counter: An instance of TokenCountingHandler for counting tokens.
        callback_manager: An instance of CallbackManager for managing callbacks.
        qa_prompt: A string representing the chat prompt.
    """

    def __init__(self, config: ChatConfig):
        """Initializes the Chat instance.

        Args:
            config: An instance of ChatConfig containing configuration settings.
        """
        self.config = config
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            job_type="chat",
        )
        self.run._label(repo="wandbot")
        self.chat_table = StreamTable(
            f"{self.config.wandb_entity}/{self.config.wandb_project}/chat_logs"
        )

        self.wandb_callback = WandbCallbackHandler()
        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager(
            [self.wandb_callback, self.token_counter]
        )
        self.default_service_context = load_service_context(
            llm=self.config.chat_model_name,
            temperature=self.config.chat_temperature,
            max_retries=self.config.max_retries,
            embeddings_cache=str(self.config.embeddings_cache),
            callback_manager=self.callback_manager,
        )
        self.fallback_service_context = load_service_context(
            llm=self.config.fallback_model_name,
            temperature=self.config.chat_temperature,
            max_retries=self.config.max_fallback_retries,
            embeddings_cache=str(self.config.embeddings_cache),
            callback_manager=self.callback_manager,
        )

        self.qa_prompt = load_chat_prompt(
            f_name=self.config.chat_prompt,
            system_template=self.config.system_template,
            # language_code=language,
            # query_intent=kwargs.get("query_intent", ""),
        )
        self.query_handler = QueryHandler()
        self.retriever = Retriever(
            run=self.run,
            service_context=self.fallback_service_context,
            callback_manager=self.callback_manager,
        )

    def _load_chat_engine(
        self,
        service_context: ServiceContext,
        has_chat_history: bool = False,
        language: str = "en",
        initial_k: int = 15,
        top_k: int = 5,
        **kwargs,
    ) -> BaseChatEngine:
        """Loads the chat engine with the given model name and maximum retries.

        Args:
            model_name: A string representing the name of the model.
            max_retries: An integer representing the maximum number of retries.

        Returns:
            An instance of ChatEngine.
        """

        query_engine = self.retriever.load_query_engine(
            similarity_top_k=initial_k,
            language=language,
            top_k=top_k,
        )

        self.qa_prompt = load_chat_prompt(
            f_name=self.config.chat_prompt,
            system_template=self.config.system_template,
            language_code=language,
            query_intent=kwargs.get("query_intent", ""),
        )
        chat_engine_kwargs = dict(
            retriever=query_engine.retriever,
            storage_context=self.retriever.storage_context,
            service_context=service_context,
            text_qa_template=self.qa_prompt,
            similarity_top_k=initial_k,
            response_mode="compact",
            node_postprocessors=[
                LanguageFilterPostprocessor(languages=[language, "python"]),
                CohereRerank(top_n=top_k, model="rerank-english-v2.0")
                if language == "en"
                else CohereRerank(
                    top_n=top_k, model="rerank-multilingual-v2.0"
                ),
            ],
        )

        if has_chat_history:
            chat_engine = CondensePlusContextChatEngine.from_defaults(
                **chat_engine_kwargs
            )
        else:
            chat_engine = ContextChatEngine.from_defaults(**chat_engine_kwargs)

        return chat_engine

    def format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the response dictionary.

        Args:
            result: A dictionary representing the response.

        Returns:
            A formatted response dictionary.
        """
        response = {}
        if result.get("source_documents", None):
            source_documents = [
                {
                    "source": doc.metadata["source"],
                    "text": doc.text,
                }
                for doc in result["source_documents"]
            ]
        else:
            source_documents = []
        response["answer"] = result["answer"]
        response["model"] = result["model"]

        if len(source_documents) and self.config.include_sources:
            response["source_documents"] = json.dumps(source_documents)
            response["sources"] = ",".join(
                [doc["source"] for doc in source_documents]
            )
        else:
            response["source_documents"] = ""
            response["sources"] = ""

        return response

    def get_response(
        self,
        service_context: ServiceContext,
        query: str,
        language: str,
        chat_history: List[ChatMessage],
        **kwargs,
    ) -> Dict[str, Any]:
        chat_engine = self._load_chat_engine(
            service_context=service_context,
            language=language,
            has_chat_history=bool(chat_history),
            **kwargs,
        )
        response = chat_engine.chat(message=query, chat_history=chat_history)
        result = {
            "answer": response.response,
            "source_documents": response.source_nodes,
            "model": self.config.chat_model_name,
        }
        return result

    def get_answer(
        self,
        resolved_query: ResolvedQuery,
        **kwargs,
    ) -> Dict[str, Any]:
        """Gets the answer for the given query and chat history.

        Args:
            resolved_query: An instance of ResolvedQuery representing the resolved query.

        Returns:
            A dictionary representing the answer.

        """
        try:
            result = self.get_response(
                service_context=self.default_service_context,
                query=resolved_query.cleaned_query,
                language=resolved_query.language,
                chat_history=resolved_query.chat_history,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"{self.config.chat_model_name} failed with {e}")
            logger.warning(
                f"Falling back to {self.config.fallback_model_name} model"
            )
            try:
                result = self.get_response(
                    service_context=self.fallback_service_context,
                    query=resolved_query.cleaned_query,
                    language=resolved_query.language,
                    chat_history=resolved_query.chat_history,
                    **kwargs,
                )

            except Exception as e:
                logger.error(
                    f"{self.config.fallback_model_name} failed with {e}"
                )
                result = {
                    "answer": "\uE058"
                    + " Sorry, there seems to be an issue with our LLM service. Please try again in some time.",
                    "source_documents": None,
                    "model": "None",
                }
        return self.format_response(result)

    def __call__(self, chat_request: ChatRequest) -> ChatResponse:
        """Handles the chat request and returns the chat response.

        Args:
            chat_request: An instance of ChatRequest representing the chat request.

        Returns:
            An instance of `ChatResponse` representing the chat response.
        """
        with Timer() as timer:
            try:
                resolved_query = self.query_handler(chat_request)
            except ValueError as e:
                result = {
                    "answer": str(e),
                    "sources": "",
                }
            else:
                result = self.get_answer(resolved_query)
                usage_stats = {
                    "total_tokens": self.token_counter.total_llm_token_count,
                    "prompt_tokens": self.token_counter.prompt_llm_token_count,
                    "completion_tokens": self.token_counter.completion_llm_token_count,
                }
                self.token_counter.reset_counts()

        result.update(
            dict(
                **{
                    "question": chat_request.question,
                    "time_taken": timer.elapsed,
                    "start_time": timer.start,
                    "end_time": timer.stop,
                    "application": chat_request.application,
                },
                **usage_stats,
            )
        )
        self.run.log(usage_stats)
        result["system_prompt"] = self.qa_prompt.message_templates[0].content
        self.chat_table.log(result)
        return ChatResponse(**result)
