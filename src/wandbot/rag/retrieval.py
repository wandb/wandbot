import os
import logging
from typing import Any, Dict, List
from copy import deepcopy

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import BaseModel

from wandbot.rag.utils import get_web_contexts
from wandbot.retriever.base import VectorStore
from wandbot.retriever.web_search import YouSearch, YouSearchConfig

import cohere
import weave
from weave.trace.autopatch import autopatch
from weave.integrations.langchain.langchain import langchain_patcher

logger = logging.getLogger(__name__)

autopatch()
langchain_patcher.undo_patch()

class WebSearchResults(BaseModel):
    web_search_success: bool
    web_contexts: List

co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

@weave.op
def reciprocal_rank_fusion(results: list[list[Document]], smoothing_constant=60):
    """Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Implements the RRF algorithm from Cormack et al. (2009) to fuse multiple 
    ranked lists into a single ranked list. Documents appearing in multiple
    lists have their reciprocal rank scores summed.
    
    Args:
        results: List of ranked document lists to combine
        smoothing_constant: Constant that controls scoring impact (default: 60). 
            It smooths out the differences between ranks by adding a constant to 
            the denominator in the formula 1/(rank + k). This prevents very high 
            ranks (especially rank 1) from completely dominating the fusion results.
    
    Returns:
        List[Document]: Combined and reranked list of documents
    """
    text_to_doc = {}
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_content = doc.page_content
            text_to_doc[doc_content] = doc
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0.0
            fused_scores[doc_content] += 1 / (rank + smoothing_constant)

    ranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    ranked_results = [text_to_doc[text] for text in ranked_results.keys()]
    return ranked_results


@weave.op
def run_web_search(query, avoid=False) -> WebSearchResults:
    try:
        if avoid:
            logger.debug(f"Skipping web search, avoid: {avoid}")
            return WebSearchResults(
                web_search_success=False,
                web_contexts=[],
            )
        yousearch = YouSearch(YouSearchConfig())
        web_results = yousearch(query)
        if web_results.success:
            web_contexts = get_web_contexts(web_results)
        else:
            logger.debug(
                f"Issue running web search, web_results: {web_results}"
            )
            web_contexts = []
        return WebSearchResults(
            web_search_success=web_results.success,
            web_contexts=web_contexts,
        )
    except Exception as e:
        logger.error(f"Error running web search: {e}")
        return WebSearchResults(
            web_search_success=False,
            web_contexts=[],
        )


class FusionRetrieval:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int,
        search_type: str,
        english_reranker_model: str,
        multilingual_reranker_model: str,
    ):
        self.vectorstore = vector_store
        self.top_k = top_k
        self.search_type = search_type

        self.retriever = self.vectorstore.as_retriever(
            search_type=self.search_type, search_kwargs={"k": self.top_k}
        )

        self._chain = None
        self.english_reranker_model = english_reranker_model
        self.multilingual_reranker_model = multilingual_reranker_model

    @weave.op
    def rerank_results(
        self,
        queries: List[str],
        context: List[Document],
        top_k: int,
        language: str = "en",
    ) -> List[Document]:
        
        query = "\n".join(queries)
        documents = [doc.page_content for doc in context]
        reranker_model_name = (
            self.english_reranker_model if language == "en" 
            else self.multilingual_reranker_model
        )
        
        results = co.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model=reranker_model_name
        )
        
        reranked_docs = []
        for hit in results.results:
            original_doc = context[hit.index]
            doc_copy = Document(
                page_content=original_doc.page_content,
                metadata=deepcopy(original_doc.metadata)
            )
            doc_copy.metadata["relevance_score"] = hit.relevance_score
            reranked_docs.append(doc_copy)
            
        return reranked_docs

    @weave.op
    def retriever_batch(self, queries):
        """wrapped for weave tracking"""
        return self.retriever.batch(queries)

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = (
                RunnablePassthrough().assign(
                    docs_context=lambda x: self.retriever_batch(
                        x["all_queries"]
                    ),
                    search_results=lambda x: run_web_search(
                        query=x["standalone_query"],
                        avoid=True  # Always skip web search
                        # x["avoid_query"]
                    ),
                )
                | RunnablePassthrough().assign(
                    full_context=lambda x: x["docs_context"]
                    + [x["search_results"].web_contexts],
                    web_search_success=lambda x: x[
                        "search_results"
                    ].web_search_success,
                )
                | RunnablePassthrough().assign(
                    fused_context=lambda x: reciprocal_rank_fusion(
                        x["full_context"]
                    )
                )
                | RunnablePassthrough().assign(
                    context=lambda x: self.rerank_results(
                        [x["standalone_query"]],
                        x["fused_context"],
                        self.top_k,
                        x["language"],
                    )
                )
            )
        return self._chain

    @weave.op
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.chain.invoke(inputs)
