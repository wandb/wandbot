"""Handles chat interactions for WandBot.

This module contains the Chat class which is responsible for handling chat interactions with the WandBot system.
It provides both synchronous and asynchronous interfaces for chat operations, manages translations between
languages, and coordinates the RAG (Retrieval Augmented Generation) pipeline.

The Chat class handles:
- Initialization of the vector store and RAG pipeline
- Translation between Japanese and English (when needed)
- Error handling and status tracking
- Timing of operations
- Response generation and formatting

Typical usage example:

    from wandbot.configs.chat_config import ChatConfig
    from wandbot.configs.vector_store_config import VectorStoreConfig
    from wandbot.chat.schemas import ChatRequest
    
    # Initialize with both required configs
    vector_store_config = VectorStoreConfig()
    chat_config = ChatConfig()
    chat = Chat(vector_store_config=vector_store_config, chat_config=chat_config)
    
    # Async usage
    async def chat_example():
        response = await chat.__acall__(
            ChatRequest(
                question="How do I use wandb?",
                chat_history=[],
                language="en"
            )
        )
        print(f"Answer: {response.answer}")
        print(f"Time taken: {response.time_taken}")
    
    # Sync usage
    response = chat(
        ChatRequest(
            question="How do I use wandb?",
            chat_history=[],
            language="en"
        )
    )
    print(f"Answer: {response.answer}")
    print(f"Time taken: {response.time_taken}")
"""
import sys
import traceback
from typing import List

import weave

from wandbot.chat.rag import RAGPipeline, RAGPipelineOutput
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.chat.utils import translate_en_to_ja, translate_ja_to_en
from wandbot.configs.chat_config import ChatConfig
from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.database.schemas import QuestionAnswer
from wandbot.retriever import VectorStore
from wandbot.utils import ErrorInfo, Timer, get_error_file_path, get_logger, run_sync

logger = get_logger(__name__)

class Chat:
    """Class for handling chat interactions and managing the RAG system components."""

    def __init__(self, vector_store_config: VectorStoreConfig, chat_config: ChatConfig):
        """Initializes the Chat instance with all necessary RAG components.
        
        Args:
            vector_store_config: Configuration for vector store setup
            chat_config: Configuration for chat and RAG behavior
        """
        self.chat_config = chat_config
        
        # Initialize vector store internally
        self.vector_store = VectorStore.from_config(
            vector_store_config=vector_store_config,
            chat_config=chat_config
        )
        
        # Initialize RAG pipeline with internal vector store
        self.rag_pipeline = RAGPipeline(
            vector_store=self.vector_store,
            chat_config=chat_config,
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
        
        # Initialize working request with original request
        working_request = chat_request
        
        with Timer() as timer:
            try:
                # Handle Japanese translation
                if original_language == "ja":
                    try:
                        translated_question = translate_ja_to_en(
                            chat_request.question,
                            self.chat_config.ja_translation_model_name
                        )
                        working_request = ChatRequest(
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
                        api_call_statuses["chat_success"] = False
                        api_call_statuses["chat_error_info"] = error_info.model_dump()
                        # Create error response preserving translation error context
                        return ChatResponse(
                            system_prompt="",
                            question=chat_request.question,  # Original question
                            answer=f"Translation error: {str(e)}",
                            response_synthesis_llm_messages=[],
                            model="",
                            sources="",
                            source_documents="",
                            total_tokens=0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            time_taken=timer.elapsed,
                            start_time=timer.start,
                            end_time=timer.stop,
                            application=chat_request.application,
                            api_call_statuses=api_call_statuses
                        )

                # Get answer using working request
                result = await self._aget_answer(
                    working_request.question, working_request.chat_history or []
                )

                result_dict = result.model_dump()
                api_call_statuses.update(result_dict.get("api_call_statuses", {}))

                # Handle Japanese translation of response
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
                        api_call_statuses["chat_success"] = False
                        api_call_statuses["chat_error_info"] = error_info.model_dump()
                        # Return response with translation error but preserve original answer
                        result_dict["answer"] = f"Translation error: {str(e)}\nOriginal answer: {result_dict['answer']}"

                # Update with final metadata
                api_call_statuses["chat_success"] = True
                api_call_statuses["chat_error_info"] = ErrorInfo(
                    has_error=False,
                    error_message="",
                    error_type="",
                    stacktrace="",
                    file_path="",
                    component="chat"
                ).model_dump()
                result_dict.update({
                    "application": chat_request.application,
                    "api_call_statuses": api_call_statuses,
                    "time_taken": timer.elapsed,
                    "start_time": timer.start,
                    "end_time": timer.stop,
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
                api_call_statuses["chat_success"] = False
                api_call_statuses["chat_error_info"] = error_info.model_dump()
                
                return ChatResponse(
                    system_prompt="",
                    question=chat_request.question,
                    answer=error_info.error_message,
                    response_synthesis_llm_messages=[],
                    model="",
                    sources="",
                    source_documents="",
                    total_tokens=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    time_taken=timer.elapsed,
                    start_time=timer.start,
                    end_time=timer.stop,
                    application=chat_request.application,
                    api_call_statuses=api_call_statuses
                )

    @weave.op
    def __call__(self, chat_request: ChatRequest) -> ChatResponse:
        return run_sync(self.__acall__(chat_request))
