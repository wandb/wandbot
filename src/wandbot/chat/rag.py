import datetime
from typing import List, Tuple

import weave
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

from wandbot.rag.query_handler import QueryEnhancer
from wandbot.configs.chat_config import ChatConfig
from wandbot.rag.response_synthesis import ResponseSynthesizer
from wandbot.rag.retrieval import FusionRetrievalEngine
from wandbot.retriever import VectorStore
from wandbot.utils import Timer, get_logger, run_sync

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
        )
        self.retrieval_engine = FusionRetrievalEngine(
            vector_store=vector_store,
            top_k=chat_config.top_k,
            search_type=chat_config.search_type,
            english_reranker_model=chat_config.english_reranker_model,
            multilingual_reranker_model=chat_config.multilingual_reranker_model,
            do_web_search=chat_config.do_web_search,
        )
        self.response_synthesizer = ResponseSynthesizer(
            model=chat_config.response_synthesizer_model,
            temperature=chat_config.response_synthesizer_temperature,
            fallback_model=chat_config.response_synthesizer_fallback_model,
            fallback_temperature=chat_config.response_synthesizer_fallback_temperature,
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

        # with get_openai_callback() as query_enhancer_cb, Timer() as query_enhancer_tb:
        # If QueryEnhancer is fully sync, do:
        enhanced_query = self.query_enhancer({"query": question, "chat_history": chat_history})
        # or if it is truly async, do:
        # enhanced_query = await self.query_enhancer.acall({"query": question, "chat_history": chat_history})

        # with Timer() as retrieval_tb:
        # If retrieval_engine is async, do:
        retrieval_results = await self.retrieval_engine.__acall__(enhanced_query)

        # with get_openai_callback() as response_cb, Timer() as response_tb:
        response = self.response_synthesizer(retrieval_results)
        # or if it is truly async, do:
        # response = await self.response_synthesizer.__acall__(retrieval_results)

        # Build final output
        output = RAGPipelineOutput(
            question=enhanced_query["standalone_query"],
            answer=response["response"],
            sources="\n".join(
                [item.metadata["source"] for item in retrieval_results["context"]]
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
            api_call_statuses={"web_search_success": retrieval_results["web_search_success"]},
        )
        return output

    @weave.op
    def __call__(
         self, question: str, chat_history: List[Tuple[str, str]] | None = None
     ) -> RAGPipelineOutput:
        return run_sync(self.__acall__(question, chat_history))