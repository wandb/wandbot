import datetime
from typing import List, Tuple

import weave
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

from wandbot.rag.query_handler import QueryEnhancer
from wandbot.rag.response_synthesis import ResponseSynthesizer
from wandbot.rag.retrieval import FusionRetrieval
from wandbot.retriever import VectorStore
from wandbot.utils import Timer, get_logger

logger = get_logger(__name__)


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
        top_k: int = 15,
        search_type: str = "mmr",
        english_reranker_model: str = "rerank-english-v2.0",
        multilingual_reranker_model: str = "rerank-multilingual-v2.0",
        response_synthesizer_model: str = "gpt-4-0125-preview",
        response_synthesizer_temperature: float = 0.1,
        response_synthesizer_fallback_model: str = "gpt-4-0125-preview",
        response_synthesizer_fallback_temperature: float = 0.1,
    ):
        self.vector_store = vector_store
        self.query_enhancer = QueryEnhancer()
        self.retrieval = FusionRetrieval(
            vector_store=vector_store, 
            top_k=top_k, 
            search_type=search_type,
            english_reranker_model=english_reranker_model,
            multilingual_reranker_model=multilingual_reranker_model,
        )
        self.response_synthesizer = ResponseSynthesizer(
            model=response_synthesizer_model,
            temperature=response_synthesizer_temperature,
            fallback_model=response_synthesizer_fallback_model,
            fallback_temperature=response_synthesizer_fallback_temperature,
        )

    @weave.op()
    def __call__(
        self, question: str, chat_history: List[Tuple[str, str]] | None = None
    ) -> RAGPipelineOutput:
        if chat_history is None:
            chat_history = []

        with get_openai_callback() as query_enhancer_cb, Timer() as query_enhancer_tb:
            enhanced_query = self.query_enhancer(
                {"query": question, "chat_history": chat_history}
            )

        with Timer() as retrieval_tb:
            retrieval_results = self.retrieval(enhanced_query)
            logger.debug(f"Retrieval results: {retrieval_results}")

        with get_openai_callback() as response_cb, Timer() as response_tb:
            response = self.response_synthesizer(retrieval_results)

        output = RAGPipelineOutput(
            question=enhanced_query["standalone_query"],
            answer=response["response"],
            sources="\n".join(
                [
                    item.metadata["source"]
                    for item in retrieval_results["context"]
                ]
            ),
            source_documents=response["context_str"],
            system_prompt=response["response_prompt"],
            model=response["response_model"],
            total_tokens=query_enhancer_cb.total_tokens
            + response_cb.total_tokens,
            prompt_tokens=query_enhancer_cb.prompt_tokens
            + response_cb.prompt_tokens,
            completion_tokens=query_enhancer_cb.completion_tokens
            + response_cb.completion_tokens,
            time_taken=query_enhancer_tb.elapsed
            + retrieval_tb.elapsed
            + response_tb.elapsed,
            start_time=query_enhancer_tb.start,
            end_time=response_tb.stop,
            api_call_statuses={
                "web_search_success": retrieval_results["web_search_success"],
            },
        )

        return output
