import os
from typing import List, Optional

import requests
from llama_index import QueryBundle
from llama_index.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.base_retriever import BaseRetriever
from llama_index.schema import NodeWithScore, QueryType, TextNode
from wandbot.utils import get_logger

logger = get_logger(__name__)


class YouRetriever(BaseRetriever):
    """You retriever."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        similarity_top_k: int = 10,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""
        self._api_key = api_key or os.environ["YOU_API_KEY"]
        self.similarity_top_k = (
            similarity_top_k if similarity_top_k <= 20 else 20
        )
        super().__init__(callback_manager)

    def _retrieve(
        self, query_bundle: QueryBundle, **kwargs
    ) -> List[NodeWithScore]:
        """Retrieve."""
        if kwargs.get("is_avoid_query", False):
            try:
                headers = {"X-API-Key": self._api_key}
                url = "https://api.ydc-index.io/search"

                querystring = {
                    "query": "Weights & Biases, W&B, wandb or Weave "
                    + query_bundle.query_str,
                    "num_web_results": self.similarity_top_k,
                }
                response = requests.get(
                    url, headers=headers, params=querystring
                )
                if response.status_code != 200:
                    return []
                else:
                    results = response.json()

                snippets = [hit["snippets"] for hit in results["hits"]]
                snippet_metadata = [
                    {
                        "source": hit["url"],
                        "language": "en",
                        "description": hit["description"],
                        "title": hit["title"],
                        "tags": ["you.com"],
                    }
                    for hit in results["hits"]
                ]
                search_hits = []
                for snippet_list, metadata in zip(snippets, snippet_metadata):
                    for snippet in snippet_list:
                        search_hits.append((snippet, metadata))

                return [
                    NodeWithScore(
                        node=TextNode(text=s[0], metadata=s[1]),
                        score=1.0,
                    )
                    for s in search_hits
                ]
            except Exception as e:
                return []
        else:
            return []

    async def _aretrieve(
        self, query_bundle: QueryBundle, **kwargs
    ) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return self._retrieve(query_bundle, **kwargs)

    def retrieve(
        self, str_or_query_bundle: QueryType, **kwargs
    ) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Args:
            str_or_query_bundle (QueryType): Either a query string or
                a QueryBundle object.

        """
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
                nodes = self._handle_recursive_retrieval(query_bundle, nodes)
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
