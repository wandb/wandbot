import asyncio
import os
import logging
from typing import Any, Dict, List
from copy import deepcopy
import traceback
import sys

from wandbot.schema.document import Document
from wandbot.schema.retrieval import RetrievalResult
from wandbot.schema.api_status import APIStatus
from wandbot.configs.chat_config import ChatConfig
import cohere
import weave
from weave.trace.autopatch import autopatch
from wandbot.utils import run_sync, get_error_file_path, ErrorInfo
from tenacity import retry, stop_after_attempt, wait_exponential, after_log

from wandbot.retriever.base import VectorStore
from wandbot.retriever.web_search import _async_run_web_search

logger = logging.getLogger(__name__)
retry_chat_config = ChatConfig()

autopatch()

class FusionRetrievalEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        chat_config: ChatConfig,
    ):
        self.vectorstore = vector_store
        self.chat_config = chat_config
        try:
            self.reranker_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        except Exception as e:
            logger.error(f"FUSION-RETRIEVAL: Issue with initialising rerank api:\n{e}\n")
            raise e

    @retry(
        stop=stop_after_attempt(retry_chat_config.reranker_max_retries), 
        wait=wait_exponential(multiplier=retry_chat_config.reranker_retry_multiplier,
                               min=retry_chat_config.reranker_retry_min_wait, 
                               max=retry_chat_config.reranker_retry_max_wait),
        before_sleep=lambda retry_state: logger.warning(
            f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
        ),
        after=after_log(logger, logging.ERROR)
    )
    @weave.op
    def rerank_results(
        self,
        query: str,
        context: List[Document],
        top_k: int,
        language: str = "en",
    ) -> tuple[List[Document], APIStatus]:
        """Reranks results and returns both results and API status"""
        api_status = APIStatus(component="reranker_api", success=True)
        
        documents = [doc.page_content for doc in context]
        reranker_model_name = (
            self.chat_config.english_reranker_model if language == "en" 
            else self.chat_config.multilingual_reranker_model
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
            error = f"FUSION-RETRIEVAL: Issue with rerank api:\n{e}\n"
            logger.error(error)
            error_info = ErrorInfo(
                component="reranker_api",
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2])
            )
            return [], APIStatus(
                component="reranker_api",
                success=False,
                error_info=error_info
            )
        
        reranked_docs = []
        for hit in results.results:
            original_doc = context[hit.index]
            doc_copy = Document(
                page_content=original_doc.page_content,
                metadata=deepcopy(original_doc.metadata)
            )
            doc_copy.metadata["reranker_relevance_score"] = hit.relevance_score
            reranked_docs.append(doc_copy)
            
        return reranked_docs, api_status

    async def _async_rerank_results(
        self,
        query: str,
        context: List[Document],
        top_k: int,
        language: str = "en",
    ) -> tuple[List[Document], APIStatus]:
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

    @weave.op
    async def _run_retrieval_common(self, inputs: Dict[str, Any], use_async: bool) -> Dict[str, Any]:
        """Single function containing the entire retrieval logic."""
        try:
            if use_async:
                docs_context, web_search_results = await asyncio.gather(
                    self.vectorstore._async_retrieve(
                        query_texts=inputs["all_queries"],
                    ),
                    _async_run_web_search(
                        query=inputs["standalone_query"],
                        top_k=self.chat_config.top_k,
                        avoid=not self.chat_config.do_web_search
                    )
                )
            else:
                docs_context = self.vectorstore.retrieve(
                    query_texts=inputs["all_queries"],
                )
                web_search_results = run_sync(_async_run_web_search(
                    query=inputs["standalone_query"],
                    top_k=self.chat_config.top_k,
                    avoid=not self.chat_config.do_web_search
                ))

            docs_context, embedding_status = self._flatten_retrieved_results(docs_context)

            logger.debug(f"RETRIEVAL-ENGINE: First retrieved document from vector store:\n{docs_context[0]}\n")
            logger.info(f"RETRIEVAL-ENGINE: Retrieved {len(docs_context)} documents from vector store.")
            if self.chat_config.do_web_search:  
                logger.info(f"RETRIEVAL-ENGINE: Retrieved {len(web_search_results.web_contexts)} web contexts.")
            
            fused_context = docs_context + web_search_results.web_contexts

            # Dedupe results
            len_fused_context = len(fused_context)
            fused_context_deduped = self.dedupe_retrieved_results(fused_context)
            logger.info(f"RETRIEVAL-ENGINE: Deduped {len_fused_context - len(fused_context_deduped)} duplicate documents.")
            
            # Rerank results
            try:
                if use_async:
                    context, api_status = await self._async_rerank_results(
                        query=inputs["standalone_query"],
                        context=fused_context_deduped,
                        top_k=self.chat_config.top_k,
                        language=inputs["language"]
                    )
                else:
                    context, api_status = self.rerank_results(
                        query=inputs["standalone_query"],
                        context=fused_context_deduped,
                        top_k=self.chat_config.top_k,
                        language=inputs["language"]
                    )

                if not api_status.success:
                    err_msg = f"FUSION-RETRIEVAL: Reranker failed: {api_status.component}, \
{api_status.error_info.model_dump_json(indent=4)}"
                    logger.error(err_msg)
                    context = [f"Error: {err_msg}"]
                    raise Exception(err_msg)  # Raise for weave tracing
            except Exception as e:
                error_info = ErrorInfo(
                    component="reranker_api",
                    has_error=True,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    stacktrace=''.join(traceback.format_exc()),
                    file_path=get_error_file_path(sys.exc_info()[2])
                )
                err_msg = f"FUSION-RETRIEVAL: Reranker failed: {error_info.component}, \
{error_info.model_dump_json(indent=4)}"
                logger.error(err_msg)
                context = [f"Error: {err_msg}"]  # Fallback to non-reranked results
                raise Exception(err_msg)  # Raise for weave tracing
                
            logger.debug(f"RETRIEVAL-ENGINE: Reranked and selected {len(fused_context_deduped)} -> {len(context)} documents.")
            
            # Return retrieval result
            retrieval_result = RetrievalResult(
                documents=context,
                retrieval_info={
                    "query": inputs["standalone_query"],
                    "language": inputs["language"],
                    "intents": inputs.get("intents", []),
                    "sub_queries": inputs.get("sub_queries", []),
                    "num_vector_store_docs": len(docs_context),
                    "num_web_docs": len(web_search_results.web_contexts),
                    "num_deduped": len_fused_context - len(fused_context_deduped),
                    "api_statuses": {
                        "web_search_api": web_search_results.api_status,
                        "reranker_api": api_status,
                        "embedding_api": embedding_status
                    }
                }
            )
            
            return retrieval_result
        except Exception as e:
            logger.error(f"FUSION-RETRIEVAL: Error in retrieval: {e}")
            raise

    @weave.op
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return run_sync(self._run_retrieval_common(inputs, use_async=False))

    async def __acall__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self._run_retrieval_common(inputs, use_async=True)
    
    def _flatten_retrieved_results(self, results: Dict[str, Any]) -> tuple[List[Document], Any]:
        embedding_status = results.get("_embedding_status")
        docs = [results[k] for k in results.keys() if not k.startswith('_')]  # Skip metadata keys
        if isinstance(docs, list) and all(isinstance(d, list) for d in docs):
            docs = [item for sublist in docs for item in sublist]  # flattens to List[Tuple[Document, float]]
            # Convert tuples to Documents if needed
            processed_docs = []
            for item in docs:
                if isinstance(item, Document):
                    processed_docs.append(item)
                elif isinstance(item, tuple) and len(item) == 2:
                    doc, score = item
                    if isinstance(doc, Document):
                        if 'relevance_score' not in doc.metadata:
                            doc.metadata['relevance_score'] = score
                        processed_docs.append(doc)
            return processed_docs, embedding_status
        return docs if isinstance(docs, list) else [docs], embedding_status

