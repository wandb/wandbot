from typing import List, Tuple

from langchain_community.callbacks import get_openai_callback
from wandbot.ingestion.config import VectorStoreConfig
from wandbot.rag import FusionRetrieval, QueryEnhancer, ResponseSynthesizer
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


class Pipeline:
    def __init__(
        self,
        vector_store_config: VectorStoreConfig,
        top_k: int = 15,
        search_type: str = "mmr",
    ):
        self.query_enhancer = QueryEnhancer()
        self.retrieval = FusionRetrieval(
            vector_store_config, top_k=top_k, search_type=search_type
        )
        self.response_synthesizer = ResponseSynthesizer()

    def __call__(
        self, question: str, chat_history: List[Tuple[str, str]] | None = None
    ):
        with get_openai_callback() as query_enhancer_cb, Timer() as query_enhancer_tb:
            enhanced_query = self.query_enhancer.chain.invoke(
                {"query": question, "chat_history": chat_history}
            )
        with get_openai_callback() as retrieval_cb, Timer() as retrieval_tb:
            retrieval_results = self.retrieval.chain.invoke(enhanced_query)
        with get_openai_callback() as response_cb, Timer() as response_tb:
            response = self.response_synthesizer.chain.invoke(
                {"query": enhanced_query, "context": retrieval_results}
            )

        contexts = {
            "context": [
                {"page_content": item.page_content, "metadata": item.metadata}
                for item in retrieval_results
            ]
        }

        return {
            "enhanced_query": {
                **enhanced_query,
                **get_stats_dict_from_token_callback(query_enhancer_cb),
                **get_stats_dict_from_timer(query_enhancer_tb),
            },
            "retrieval_results": {
                **contexts,
                **get_stats_dict_from_token_callback(retrieval_cb),
                **get_stats_dict_from_timer(retrieval_tb),
            },
            "response": {
                **response,
                **get_stats_dict_from_token_callback(response_cb),
                **get_stats_dict_from_timer(response_tb),
            },
        }
