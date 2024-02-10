import os
from typing import List, Union

from langchain.load import dumps, loads
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document, format_document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from wandbot.retriever.external import YouRetriever
from wandbot.utils import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        index: Union[VectorStoreIndex, BaseIndex],
        storage_context,
        similarity_top_k: int,
        language: str,
        indices: List[str],
        include_tags: List[str],
        include_web_results: bool,
    ):
        self.index = index
        self.storage_context = storage_context

        self._filters = self._load_filters(
            language=language,
            indices=indices,
            include_tags=include_tags,
        )
        self.include_web_results = include_web_results

        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k,
            storage_context=self.storage_context,
            filters=self._filters,
        )

        self.you_retriever = YouRetriever(
            api_key=os.environ.get("YOU_API_KEY"),
            similarity_top_k=similarity_top_k,
        )
        super().__init__()

    def _load_filters(
        self, language: str, indices: List[str], include_tags
    ) -> MetadataFilters:
        index_filters = [
            ExactMatchFilter(key="index", value=idx) for idx in indices
        ]
        language_filter = [
            ExactMatchFilter(key="language", value=lang)
            for lang in [language, "python"]
        ]
        include_tags_filter = [
            MetadataFilter(
                key="tags", value=tag, operator=FilterOperator.TEXT_MATCH
            )
            for tag in include_tags
        ]

        filters = index_filters + language_filter + include_tags_filter

        metadata_filters = MetadataFilters(
            filters=filters,
            condition=FilterCondition.OR,
        )
        return metadata_filters

    def _retrieve(self, query: QueryBundle, **kwargs):
        vector_nodes = self.vector_retriever.retrieve(query)
        you_nodes = []
        if self.include_web_results:
            you_nodes = self.you_retriever.retrieve(query)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in vector_nodes + you_nodes:
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