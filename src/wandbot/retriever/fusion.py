import os
from typing import Dict, List, Optional, Tuple

from llama_index import QueryBundle
from llama_index.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.base_retriever import BaseRetriever
from llama_index.llms.utils import LLMType
from llama_index.retrievers import BM25Retriever, QueryFusionRetriever
from llama_index.retrievers.fusion_retriever import FUSION_MODES
from llama_index.schema import IndexNode, NodeWithScore, QueryType
from wandbot.retriever.external import YouRetriever
from wandbot.utils import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        index,
        storage_context,
        similarity_top_k: int = 20,
    ):
        self.index = index
        self.storage_context = storage_context

        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k,
            storage_context=self.storage_context,
        )
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore,
            similarity_top_k=similarity_top_k,
        )
        self.you_retriever = YouRetriever(
            api_key=os.environ.get("YOU_API_KEY"),
            similarity_top_k=similarity_top_k,
        )
        super().__init__()

    def _retrieve(self, query: QueryBundle, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query)
        vector_nodes = self.vector_retriever.retrieve(query)
        you_nodes = (
            self.you_retriever.retrieve(query)
            if not kwargs.get("is_avoid_query", False)
            else []
        )

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes + you_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

    def retrieve(
        self, str_or_query_bundle: QueryType, **kwargs
    ) -> List[NodeWithScore]:
        self._check_callback_manager()

        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle, **kwargs)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes


class FusionRetriever(QueryFusionRetriever):
    def __init__(
        self,
        retrievers: List[HybridRetriever],
        llm: Optional[LLMType] = "default",
        query_gen_prompt: Optional[str] = None,
        mode: FUSION_MODES = FUSION_MODES.SIMPLE,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        num_queries: int = 4,
        use_async: bool = True,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
    ) -> None:
        super().__init__(
            retrievers=retrievers,
            llm=llm,
            query_gen_prompt=query_gen_prompt,
            mode=mode,
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,
            use_async=use_async,
            verbose=verbose,
            callback_manager=callback_manager,
            objects=objects,
            object_map=object_map,
        )
        self._retrievers = retrievers

    def _run_sync_queries(
        self, queries: List[QueryBundle], **kwargs
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        results = {}
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                results[(query.query_str, i)] = retriever.retrieve(
                    query, **kwargs
                )

        return results

    def _retrieve(
        self, query_bundle: QueryBundle, **kwargs
    ) -> List[NodeWithScore]:
        if self.num_queries > 1:
            queries = self._get_queries(query_bundle.query_str)
        else:
            queries = [query_bundle]

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries, **kwargs)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)[
                : self.similarity_top_k
            ]
        elif self.mode == FUSION_MODES.SIMPLE:
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")

    def retrieve(
        self, str_or_query_bundle: QueryType, **kwargs
    ) -> List[NodeWithScore]:
        self._check_callback_manager()

        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle, **kwargs)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes
