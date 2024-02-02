import asyncio
from typing import Dict, List, Optional, Tuple, Union

import nest_asyncio
from llama_index import QueryBundle, VectorStoreIndex
from llama_index.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.base import BaseIndex
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.llms.utils import LLMType
from llama_index.retrievers import BM25Retriever, QueryFusionRetriever
from llama_index.retrievers.fusion_retriever import FUSION_MODES
from llama_index.schema import IndexNode, NodeWithScore, QueryType
from wandbot.retriever.external import YouRetriever
from wandbot.utils import get_logger, run_async_tasks

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        index: Union[VectorStoreIndex, BaseIndex],
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
            nodes=self.index.docstore.get_nodes(
                list(self.index.index_struct.nodes_dict.values())
            ),
            similarity_top_k=similarity_top_k,
        )

        super().__init__()

    def _retrieve(self, query: QueryBundle, **kwargs):
        nest_asyncio.apply()
        return asyncio.run(self._aretrieve(query, **kwargs))

    async def _aretrieve(self, query: QueryBundle, **kwargs):
        bm25_nodes = await self.bm25_retriever.aretrieve(query)
        vector_nodes = await self.vector_retriever.aretrieve(query)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

    def retrieve(
        self, str_or_query_bundle: QueryType, **kwargs
    ) -> List[NodeWithScore]:
        nest_asyncio.apply()
        return asyncio.run(self.aretrieve(str_or_query_bundle, **kwargs))

    async def aretrieve(
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
                nodes = await self._aretrieve(query_bundle, **kwargs)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes


class FusionRetriever(QueryFusionRetriever):
    def __init__(
        self,
        retrievers: List[
            Union[VectorIndexRetriever, BM25Retriever, YouRetriever]
        ],
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

    def _run_nested_async_queries(
        self, queries: List[QueryBundle], **kwargs
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            for i, retriever in enumerate(self._retrievers[:-1]):
                tasks.append(retriever.aretrieve(query, **kwargs))
                task_queries.append(query)

            # get you retriever results
            tasks.append(self._retrievers[-1].aretrieve(query, **kwargs))
            task_queries.append(query)

        task_results = run_async_tasks(tasks)

        results = {}
        for i, (query, query_result) in enumerate(
            zip(task_queries, task_results)
        ):
            results[(query.query_str, i)] = query_result

        return results

    def _run_sync_queries(
        self, queries: List[QueryBundle], **kwargs
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        results = {}
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                if isinstance(retriever, YouRetriever):
                    results[(query.query_str, i)] = retriever.retrieve(
                        query, **kwargs
                    )
                else:
                    results[(query.query_str, i)] = retriever.retrieve(query)

        return results

    async def _run_async_queries(
        self, queries: List[QueryBundle], **kwargs
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                tasks.append(retriever.aretrieve(query, **kwargs))
                task_queries.append(query)

        task_results = await asyncio.gather(*tasks)

        results = {}
        for i, (query, query_result) in enumerate(
            zip(task_queries, task_results)
        ):
            results[(query.query_str, i)] = query_result

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

    async def _aretrieve(
        self, query_bundle: QueryBundle, **kwargs
    ) -> List[NodeWithScore]:
        if self.num_queries > 1:
            queries = self._get_queries(query_bundle.query_str)
        else:
            queries = [query_bundle]

        results = await self._run_async_queries(queries, **kwargs)

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

    async def aretrieve(
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
                nodes = await self._aretrieve(query_bundle, **kwargs)
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle, nodes
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

        return nodes
