import datetime
from typing import List, Tuple, Dict

import weave
from pydantic import BaseModel

from wandbot.rag.query_handler import QueryEnhancer
from wandbot.configs.chat_config import ChatConfig
from wandbot.rag.response_synthesis import ResponseSynthesizer
from wandbot.rag.retrieval import FusionRetrievalEngine
from wandbot.retriever import VectorStore
from wandbot.utils import Timer, get_logger, run_sync
from wandbot.utils import ErrorInfo

logger = get_logger(__name__)
chat_config = ChatConfig()

def get_stats_dict_from_token_callback(token_callback):
    return {
        "total_tokens": token_callback.total_tokens,
        "prompt_tokens": token_callback.prompt_tokens,
        "completion_tokens": token_callback.completion_tokens,
        "successful_requests": token_callback.successful_requests,
    }


def get_stats_dict_from_timer(timer):
    return {
        "start_time": timer.start,
        "end_time": timer.stop,
        "time_taken": timer.elapsed,
    }


class RAGPipelineOutput(BaseModel):
    question: str
    answer: str
    sources: str
    source_documents: str
    system_prompt: str
    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    time_taken: float
    start_time: datetime.datetime
    end_time: datetime.datetime
    api_call_statuses: dict = {}
    response_synthesis_llm_messages: List[Dict[str, str]] | None = None


class RAGPipeline:

    def __init__(
        self,
        vector_store: VectorStore,
        chat_config: ChatConfig,
    ):
        self.vector_store = vector_store
        self.query_enhancer = QueryEnhancer(
            model_name = chat_config.query_enhancer_model,
            temperature = chat_config.query_enhancer_temperature,
            fallback_model_name = chat_config.query_enhancer_fallback_model,
            fallback_temperature = chat_config.query_enhancer_fallback_temperature,
            max_retries=chat_config.llm_max_retries
        )
        self.retrieval_engine = FusionRetrievalEngine(
            vector_store=vector_store,
            chat_config=chat_config,
        )
        self.response_synthesizer = ResponseSynthesizer(
            primary_provider=chat_config.response_synthesizer_provider,
            primary_model_name=chat_config.response_synthesizer_model,
            primary_temperature=chat_config.response_synthesizer_temperature,
            fallback_provider=chat_config.response_synthesizer_fallback_provider,
            fallback_model_name=chat_config.response_synthesizer_fallback_model,
            fallback_temperature=chat_config.response_synthesizer_fallback_temperature,
            max_retries=chat_config.llm_max_retries
        )

    async def __acall__(
        self, question: str, chat_history: List[Tuple[str, str]] | None = None
    ) -> RAGPipelineOutput:
        """
        Async version of the RAG pipeline. 
          1) query enhancement
          2) retrieval
          3) response synthesis
        """
        if chat_history is None:
            chat_history = []

        enhanced_query = await self.query_enhancer({"query": question, "chat_history": chat_history})

        # with Timer() as retrieval_tb:
        # If retrieval_engine is async, do:
        retrieval_result = await self.retrieval_engine.__acall__(enhanced_query)

        # with get_openai_callback() as response_cb, Timer() as response_tb:
        response = await self.response_synthesizer(retrieval_result)
        # or if it is truly async, do:
        # response = await self.response_synthesizer.__acall__(retrieval_results)

        # Build final output
        output = RAGPipelineOutput(
            question=enhanced_query["standalone_query"],
            answer=response["response"],
            sources="\n".join(
                [doc.metadata["source"] for doc in retrieval_result.documents]
            ),
            source_documents=response["context_str"],
            system_prompt=response["response_prompt"],
            model=response["response_model"],
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            time_taken=0,
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
            # total_tokens=query_enhancer_cb.total_tokens + response_cb.total_tokens,
            # prompt_tokens=query_enhancer_cb.prompt_tokens + response_cb.prompt_tokens,
            # completion_tokens=query_enhancer_cb.completion_tokens + response_cb.completion_tokens,
            # time_taken=query_enhancer_tb.elapsed + retrieval_tb.elapsed + response_tb.elapsed,
            # start_time=query_enhancer_tb.start,
            # end_time=response_tb.stop,
            api_call_statuses={
                "web_search_success": retrieval_result.retrieval_info["api_statuses"]["web_search_api"].success,
                "reranker_api_error_info": retrieval_result.retrieval_info["api_statuses"]["reranker_api"].error_info,
                "reranker_api_success": retrieval_result.retrieval_info["api_statuses"]["reranker_api"].success,
                "query_enhancer_llm_api_error_info": enhanced_query.get("api_statuses", {}).get("query_enhancer_llm_api", {}).error_info if enhanced_query.get("api_statuses") else None,
                "query_enhancer_llm_api_success": enhanced_query.get("api_statuses", {}).get("query_enhancer_llm_api", {}).success if enhanced_query.get("api_statuses") else False,
                "embedding_api_error_info": retrieval_result.retrieval_info["api_statuses"]["embedding_api"].error_info,
                "embedding_api_success": retrieval_result.retrieval_info["api_statuses"]["embedding_api"].success,
            },
            response_synthesis_llm_messages=response.get("response_synthesis_llm_messages")
        )
        return output

    @weave.op
    def __call__(
         self, question: str, chat_history: List[Tuple[str, str]] | None = None
     ) -> RAGPipelineOutput:
        return run_sync(self.__acall__(question, chat_history))