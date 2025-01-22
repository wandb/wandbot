import asyncio
import os
import logging
from typing import Any, Dict, List
from copy import deepcopy

from langchain_core.documents import Document
import cohere
import weave
from weave.trace.autopatch import autopatch
from wandbot.utils import run_sync

from wandbot.retriever.base import VectorStore
from wandbot.retriever.web_search import _async_run_web_search

logger = logging.getLogger(__name__)

autopatch()

class FusionRetrievalEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int,
        search_type: str,
        english_reranker_model: str,
        multilingual_reranker_model: str,
        do_web_search: bool
    ):
        self.vectorstore = vector_store
        self.top_k = top_k
        self.search_type = search_type
        self.english_reranker_model = english_reranker_model
        self.multilingual_reranker_model = multilingual_reranker_model
        self.do_web_search = do_web_search
        try:
            self.reranker_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        except Exception as e:
            logger.error(f"FUSION-RETRIEVAL: Issue with initialising rerank api:\n{e}\n")
            raise e

    @weave.op
    def rerank_results(
        self,
        query: str,
        context: List[Document],
        top_k: int,
        language: str = "en",
    ) -> List[Document]:
        
        documents = [doc.page_content for doc in context]
        reranker_model_name = (
            self.english_reranker_model if language == "en" 
            else self.multilingual_reranker_model
        )
        assert isinstance(query, str), "In rerank, `query` must be a string"
        assert len(documents) > 0, "No context documents passed to the re-ranker"
        logger.debug(f"Reranking {len(context)} documents for query: {query}")
        
        try:
            results = self.reranker_client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model=reranker_model_name
            )
            logger.info(f"FUSION-RETRIEVAL: Reranked {len(results.results)} documents.")
        except Exception as e:
            logger.error(f"FUSION-RETRIEVAL: Issue with rerank api:\n{e}\n")
            raise e
        
        reranked_docs = []
        for hit in results.results:
            original_doc = context[hit.index]
            doc_copy = Document(
                page_content=original_doc.page_content,
                metadata=deepcopy(original_doc.metadata)
            )
            doc_copy.metadata["reranker_relevance_score"] = hit.relevance_score
            reranked_docs.append(doc_copy)
            
        return reranked_docs

    async def _async_rerank_results(
        self,
        query: str,
        context: List[Document],
        top_k: int,
        language: str = "en",
    ) -> List[Document]:
        return await asyncio.to_thread(
            self.rerank_results,
            query=query,
            context=context,
            top_k=top_k,
            language=language
        )

    @weave.op
    def dedupe_retrieved_results(self, results: List[Document]) -> List[Document]:
        return list({doc.metadata["id"]: doc for doc in results}.values())

    async def _run_retrieval_common(self, inputs: Dict[str, Any], use_async: bool) -> Dict[str, Any]:
        """Single function containing the entire retrieval logic."""
        try:
            if use_async:
                docs_context, web_search_results = await asyncio.gather(
                    self.vectorstore._async_retrieve(
                        query_texts=inputs["all_queries"],
                        search_type=self.search_type,
                    ),
                    _async_run_web_search(
                        query=inputs["standalone_query"],
                        top_k=self.top_k,
                        avoid=not self.do_web_search
                    )
                )
            else:
                docs_context = self.vectorstore.retrieve(
                    query_texts=inputs["all_queries"],
                    search_type=self.search_type,
                )
                web_search_results = run_sync(_async_run_web_search(
                    query=inputs["standalone_query"],
                    top_k=self.top_k,
                    avoid=not self.do_web_search
                ))

            def flatten_retrieved_results(results: Dict[str, Any]) -> List[Document]:
                docs = [results[k] for k in results.keys()]
                if isinstance(docs, list) and all(isinstance(d, list) for d in docs):
                    docs = [item for sublist in docs for item in sublist]  # flattens to List[Tuple[Document, float]]
                return docs
            
            docs_context = flatten_retrieved_results(docs_context)

            logger.debug(f"RETRIEVAL-ENGINE: First retrieved document from vector store:\n{docs_context[0]}\n")
            logger.info(f"RETRIEVAL-ENGINE: Retrieved {len(docs_context)} documents from vector store.")
            logger.info(f"RETRIEVAL-ENGINE: Retrieved {len(web_search_results.web_contexts)} web contexts.")
            
            fused_context = docs_context + web_search_results.web_contexts

            # Dedupe results
            len_fused_context = len(fused_context)
            fused_context_deduped = self.dedupe_retrieved_results(fused_context)
            logger.info(f"RETRIEVAL-ENGINE: Deduped {len_fused_context - len(fused_context_deduped)} duplicate documents.")
            
            # Rerank results
            if use_async:
                context = await self._async_rerank_results(
                    query=inputs["standalone_query"],
                    context=fused_context_deduped,
                    top_k=self.top_k,
                    language=inputs["language"]
                )
            else:
                context = self.rerank_results(
                    query=inputs["standalone_query"],
                    context=fused_context_deduped,
                    top_k=self.top_k,
                    language=inputs["language"]
                )
                
            logger.debug(f"RETRIEVAL-ENGINE: Reranked {len(context)} documents.")
            
            return {
                "docs_context": docs_context,
                "search_results": web_search_results,
                "full_context": fused_context_deduped,
                "context": context,
                "web_search_success": web_search_results.web_search_success,
                "standalone_query": inputs["standalone_query"],
                "language": inputs["language"],
                "intents": inputs.get("intents", []),
                "sub_queries": inputs.get("sub_queries", [])
            }
        except Exception as e:
            logger.error(f"FUSION-RETRIEVAL: Error in retrieval: {e}")
            raise

    @weave.op
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return run_sync(self._run_retrieval_common(inputs, use_async=False))

    async def __acall__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self._run_retrieval_common(inputs, use_async=True)
