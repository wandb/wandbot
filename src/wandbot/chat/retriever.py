import os
import pathlib
from typing import List, Optional

import requests
import wandb
from llama_index import (
    QueryBundle,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.callbacks import CallbackManager
from llama_index.core import BaseRetriever
from llama_index.postprocessor import CohereRerank
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.retrievers import BM25Retriever
from llama_index.schema import NodeWithScore, TextNode
from llama_index.vector_stores import FaissVectorStore
from llama_index.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.vector_stores.types import DEFAULT_PERSIST_FNAME
from pydantic import Field
from pydantic_settings import BaseSettings
from wandbot.utils import get_logger, load_service_context

logger = get_logger(__name__)


class LanguageFilterPostprocessor(BaseNodePostprocessor):
    """Language-based Node processor."""

    languages: List[str] = ["en", "python"]
    min_result_size: int = 10

    @classmethod
    def class_name(cls) -> str:
        return "LanguageFilterPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""

        new_nodes = []
        for node in nodes:
            if node.metadata["language"] in self.languages:
                new_nodes.append(node)

        if len(new_nodes) < self.min_result_size:
            return new_nodes + nodes[: self.min_result_size - len(new_nodes)]

        return new_nodes


class MetadataPostprocessor(BaseNodePostprocessor):
    """Metadata-based Node processor."""

    @classmethod
    def class_name(cls) -> str:
        return "MetadataPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""

        for node in nodes:
            node.node.metadata = {
                k: v for k, v in node.metadata.items() if k in ["source"]
            }
        return nodes


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

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        try:
            headers = {"X-API-Key": self._api_key}
            url = "https://api.ydc-index.io/search"

            querystring = {
                "query": "Weights & Biases, W&B, wandb or Weave "
                + query_bundle.query_str,
                "num_web_results": self.similarity_top_k,
            }
            response = requests.get(url, headers=headers, params=querystring)
            if response.status_code != 200:
                return []
            else:
                results = response.json()

            search_hits = [
                (
                    "\n".join(hit["snippets"]),
                    {"source": hit["url"], "language": "en"},
                )
                for hit in results["hits"]
            ]
            return [
                NodeWithScore(
                    node=TextNode(text=s[0], metadata=s[1]),
                    score=1.0,
                )
                for s in search_hits
            ]
        except Exception as e:
            return []


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        index,
        storage_context,
        similarity_top_k=10,
    ):
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.storage_context = storage_context

        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            storage_context=self.storage_context,
        )
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore,
            similarity_top_k=self.similarity_top_k,
        )
        self.you_retriever = YouRetriever(
            api_key=os.environ.get("YOU_API_KEY"),
            similarity_top_k=self.similarity_top_k,
        )
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        you_nodes = self.you_retriever.retrieve(query)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in you_nodes + bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


class RetrieverConfig(BaseSettings):
    index_artifact: str = Field(
        "wandbot/wandbot-dev/wandbot_index:latest",
        env="WANDB_INDEX_ARTIFACT",
        validation_alias="wandb_index_artifact",
    )
    embeddings_cache: pathlib.Path = Field(
        pathlib.Path("data/cache/embeddings"), env="EMBEDDINGS_CACHE_PATH"
    )
    top_k: int = Field(
        default=5,
        env="RETRIEVER_TOP_K",
    )
    similarity_top_k: int = Field(
        default=10,
        env="RETRIEVER_SIMILARITY_TOP_K",
    )
    language: str = Field(
        default="en",
        env="RETRIEVER_LANGUAGE",
    )


class Retriever:
    def __init__(
        self,
        config: RetrieverConfig | None = None,
        run: wandb.wandb_sdk.wandb_run.Run | None = None,
        service_context: ServiceContext | None = None,
        callback_manager: CallbackManager | None = None,
    ):
        self.config = (
            config if isinstance(config, RetrieverConfig) else RetrieverConfig()
        )
        self.run = run
        self.service_context = (
            service_context
            if service_context
            else load_service_context(
                embeddings_cache=str(self.config.embeddings_cache),
                callback_manager=callback_manager,
            )
        )

        self.storage_context = self.load_storage_context_from_artifact(
            artifact_url=self.config.index_artifact
        )

        self.index = load_index_from_storage(self.storage_context)

    def load_storage_context_from_artifact(
        self, artifact_url: str
    ) -> StorageContext:
        """Loads the storage context from the given artifact URL.

        Args:
            artifact_url: A string representing the URL of the artifact.

        Returns:
            An instance of StorageContext.
        """
        artifact = self.run.use_artifact(artifact_url)
        artifact_dir = artifact.download()
        index_path = f"{artifact_dir}/{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
        storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_path(index_path),
            persist_dir=artifact_dir,
        )
        return storage_context

    def load_query_engine(
        self,
        similarity_top_k: int | None = None,
        top_k: int | None = None,
        language: str | None = None,
    ) -> RetrieverQueryEngine:
        similarity_top_k = similarity_top_k or self.config.similarity_top_k
        top_k = top_k or self.config.top_k
        language = language or self.config.language

        retriever = HybridRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
            storage_context=self.storage_context,
        )

        node_postprocessors = [
            LanguageFilterPostprocessor(languages=[language, "python"]),
            CohereRerank(top_n=top_k, model="rerank-english-v2.0")
            if language == "en"
            else CohereRerank(top_n=top_k, model="rerank-multilingual-v2.0"),
            MetadataPostprocessor(),
        ]
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=node_postprocessors,
            response_mode=ResponseMode.COMPACT,
            service_context=self.service_context,
        )
        return query_engine

    def retrieve(
        self,
        query: str,
        language: str | None = None,
        initial_k: int | None = None,
        top_k: int | None = None,
    ):
        """Retrieves the top k results from the index for the given query.

        Args:
            query: A string representing the query.
            language: A string representing the language of the query.
            initial_k: An integer representing the number of initial results to retrieve.
            top_k: An integer representing the number of top results to retrieve.

        Returns:
            A list of dictionaries representing the retrieved results.
        """
        initial_k = initial_k or self.config.similarity_top_k
        top_k = top_k or self.config.top_k
        language = language or self.config.language

        retrieval_engine = self.load_query_engine(
            initial_k,
            language,
            top_k,
        )
        query_bundle = QueryBundle(query_str=query)
        results = retrieval_engine.retrieve(query_bundle)

        outputs = [
            {
                "text": node.get_text(),
                "source": node.metadata["source"],
                "score": node.get_score(),
            }
            for node in results
        ]

        return outputs
