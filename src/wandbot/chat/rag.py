import datetime
from typing import List, Tuple

from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

import litellm

from wandbot.retriever import VectorStore
from wandbot.utils import Timer, get_logger, RAGPipelineConfig
from wandbot.rag.query_handler import QueryEnhancer
from wandbot.rag.retrieval import FusionRetrieval
from wandbot.rag.response_synthesis import ResponseSynthesizer, SimpleResponseSynthesizer
from wandbot.retriever.base import SimpleRetrievalEngine, SimpleRetrievalEngineWithRerank
from wandbot.rag.utils import LiteLLMTokenCostLogger


logger = get_logger(__name__)

litellm_token_and_cost_counter = LiteLLMTokenCostLogger()
litellm.callbacks = [litellm_token_and_cost_counter]


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
    completion_cost: float
    time_taken: float
    start_time: datetime.datetime
    end_time: datetime.datetime


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        config: RAGPipelineConfig,
        top_k: int = 5,
        search_type: str = "mmr",
    ):
        self.vector_store = vector_store
        self.config = config

        if config.query_enhancer_followed_by_rerank_fusion.enabled:
            self.query_enhancer = QueryEnhancer(config=config)
            self.retrieval = FusionRetrieval(
                vector_store=vector_store, top_k=top_k, search_type=search_type
            )
            self.response_synthesizer = ResponseSynthesizer(config=config)
        else:
            if config.retrieval_re_ranker.enabled:
                self.retrieval = SimpleRetrievalEngineWithRerank(
                    vector_store=vector_store, top_k=top_k, search_type=search_type
                )
            else:
                self.retrieval = SimpleRetrievalEngine(
                    vector_store=vector_store, top_k=top_k, search_type=search_type
                )
            self.response_synthesizer = SimpleResponseSynthesizer(config=config)

    def __call__(
        self, question: str, chat_history: List[Tuple[str, str]] | None = None
    ) -> RAGPipelineOutput: 
        if chat_history is None:
            chat_history = []

        if self.config.query_enhancer_followed_by_rerank_fusion.enabled:
            with Timer() as query_enhancer_tb:
                enhanced_query = self.query_enhancer.chain.invoke(
                    {"query": question, "chat_history": chat_history}
                )
            with Timer() as retrieval_tb:
                retrieval_results = self.retrieval.chain.invoke(enhanced_query)
        else:
            with Timer() as retrieval_tb:
                retrieval_results = self.retrieval(
                    question=question, language="en", top_k=self.retrieval.top_k, search_type=self.retrieval.search_type
                )

        with Timer() as response_tb:
            response = self.response_synthesizer.chain.invoke(retrieval_results)

        question = question if not self.config.query_enhancer_followed_by_rerank_fusion.enabled else enhanced_query["standalone_query"]
        usage_stats = litellm_token_and_cost_counter.get_totals()
        time_taken = (
            query_enhancer_tb.elapsed
            + retrieval_tb.elapsed
            + response_tb.elapsed
            if self.config.query_enhancer_followed_by_rerank_fusion.enabled
            else retrieval_tb.elapsed + response_tb.elapsed
        )
        start_time = (
            query_enhancer_tb.start
            if self.config.query_enhancer_followed_by_rerank_fusion.enabled
            else retrieval_tb.start
        )
        end_time = response_tb.stop

        if isinstance(retrieval_results, dict):
            context = retrieval_results["context"]
        else:
            context = retrieval_results.context
        sources = "\n".join(
            [
                item.metadata.get("source")
                for item in context
            ]
        )

        output = RAGPipelineOutput(
            question=question,
            answer=response["response"],
            sources=sources,
            source_documents=response["context_str"],
            system_prompt=response["response_prompt"],
            model=response["response_model"],
            total_tokens=usage_stats.total_tokens,
            prompt_tokens=usage_stats.prompt_tokens,
            completion_tokens=usage_stats.completion_tokens,
            completion_cost=usage_stats.total_cost,
            time_taken=time_taken,
            start_time=start_time,
            end_time=end_time,
        )

        return output
