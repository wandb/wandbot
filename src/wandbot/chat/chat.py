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
from typing import List

import weave
import asyncio
from wandbot.configs.chat_config import ChatConfig
from wandbot.chat.rag import RAGPipeline, RAGPipelineOutput
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.database.schemas import QuestionAnswer
from wandbot.retriever import VectorStore
from wandbot.utils import Timer, get_logger, run_sync
from wandbot.chat.utils import translate_ja_to_en, translate_en_to_ja
from wandbot.utils import ErrorInfo, get_error_file_path
import traceback
import sys

logger = get_logger(__name__)

class Chat:
    """
    Class for handling chat interactions and setting up the RAG pipeline, LLMs, etc.
    """

    def __init__(self, vector_store: VectorStore, config: ChatConfig):
        """Initializes the Chat instance.

        Args:
            config: An instance of ChatConfig containing configuration settings.
        """
        self.vector_store = vector_store
        self.chat_config = config
        self.rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            chat_config=config,
        )

    @weave.op
    async def _aget_answer(
        self, question: str, chat_history: List[QuestionAnswer]
    ) -> RAGPipelineOutput:
        history = []
        for item in chat_history:
            history.append(("user", item.question))
            history.append(("assistant", item.answer))
        result = await self.rag_pipeline.__acall__(question, history)
        return result

    @weave.op
    async def __acall__(self, chat_request: ChatRequest) -> ChatResponse:
        """Async method for chat interactions."""
        original_language = chat_request.language
        api_call_statuses = {}
        try:
            if original_language == "ja":
                try:
                    translated_question = translate_ja_to_en(
                        chat_request.question,
                        self.chat_config.ja_translation_model_name
                    )
                    chat_request.language = "en"
                    chat_request = ChatRequest(
                        question=translated_question,
                        chat_history=chat_request.chat_history,
                        application=chat_request.application,
                        language="en",
                    )
                except Exception as e:
                    error_info = ErrorInfo(
                        has_error=True,
                        error_message=str(e),
                        error_type=type(e).__name__,
                        stacktrace=''.join(traceback.format_exc()),
                        file_path=get_error_file_path(sys.exc_info()[2]),
                        component="translation"
                    )
                    api_call_statuses["translation"] = error_info.model_dump()
                    raise

            result = await self._aget_answer(
                chat_request.question, chat_request.chat_history or []
            )

            result_dict = result.model_dump()
            api_call_statuses.update(result_dict.get("api_call_statuses", {}))

            if original_language == "ja":
                try:
                    result_dict["answer"] = translate_en_to_ja(
                        result_dict["answer"],
                        self.chat_config.ja_translation_model_name
                    )
                except Exception as e:
                    error_info = ErrorInfo(
                        has_error=True,
                        error_message=str(e),
                        error_type=type(e).__name__,
                        stacktrace=''.join(traceback.format_exc()),
                        file_path=get_error_file_path(sys.exc_info()[2]),
                        component="translation"
                    )
                    api_call_statuses["translation"] = error_info.model_dump()
                    raise

            result_dict.update({
                "application": chat_request.application,
                "api_call_statuses": api_call_statuses
            })

            return ChatResponse(**result_dict)
        except Exception as e:
            error_info = ErrorInfo(
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2]),
                component="chat"
            )
            api_call_statuses["chat"] = error_info.model_dump()
            
            with Timer() as timer:
                result = {
                    "system_prompt": "",
                    "question": chat_request.question,
                    "answer": error_info.error_message,
                    "model": "",
                    "sources": "",
                    "source_documents": "",
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "api_call_statuses": api_call_statuses
                }
            result.update(
                {
                    "time_taken": timer.elapsed,
                    "start_time": timer.start,
                    "end_time": timer.stop,
                }
            )

            return ChatResponse(**result)

    @weave.op
    def __call__(self, chat_request: ChatRequest) -> ChatResponse:
        return run_sync(self.__acall__(chat_request))
