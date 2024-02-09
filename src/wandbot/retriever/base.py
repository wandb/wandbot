from typing import Any, Dict, List, Tuple

import wandb
from llama_index import (
    QueryBundle,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.callbacks import CallbackManager
from llama_index.core.base_retriever import BaseRetriever
from llama_index.postprocessor import CohereRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.vector_stores.types import DEFAULT_PERSIST_FNAME
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from wandbot.retriever.fusion import HybridRetriever
from wandbot.utils import get_logger, load_service_context, load_storage_context

logger = get_logger(__name__)


class RetrieverConfig(BaseSettings):
    index_artifact: str = Field(
        "wandbot/wandbot-dev/wandbot_index:latest",
        env="WANDB_INDEX_ARTIFACT",
        validation_alias="wandb_index_artifact",
    )
    embeddings_model: str = "text-embedding-3-small"
    embeddings_size: int = 512
    top_k: int = Field(
        default=10,
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
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
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
        self.run = (
            run
            if run
            else wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                job_type="retrieve",
            )
        )
        self.service_context = (
            service_context
            if service_context
            else load_service_context(
                embeddings_model=self.config.embeddings_model,
                embeddings_size=self.config.embeddings_dim,
                callback_manager=callback_manager,
            )
        )

        self.storage_context, indices = self.load_storage_context_from_artifact(
            artifact_url=self.config.index_artifact
        )

        self.index = load_index_from_storage(
            self.storage_context,
            service_context=self.service_context,
        )

    def load_storage_context_from_artifact(
        self, artifact_url: str
    ) -> Tuple[StorageContext, List[str]]:
        """Loads the storage context from the given artifact URL.

        Args:
            artifact_url: A string representing the URL of the artifact.

        Returns:
            An instance of StorageContext.
        """
        artifact = self.run.use_artifact(artifact_url)
        artifact_dir = artifact.download()
        index_path = f"{artifact_dir}/{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
        logger.debug(f"Loading index from {index_path}")
        storage_context = load_storage_context(persist_dir=artifact_dir)
        return storage_context, artifact.metadata["indices"]

    def load_query_engine(
        self,
        retriever: BaseRetriever | None = None,
        top_k: int | None = None,
        language: str | None = None,
    ) -> RetrieverQueryEngine:
        top_k = top_k or self.config.top_k
        language = language or self.config.language

        node_postprocessors = [
            CohereRerank(top_n=top_k, model="rerank-english-v2.0")
            if language == "en"
            else CohereRerank(top_n=top_k, model="rerank-multilingual-v2.0"),
        ]
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=node_postprocessors,
            response_mode=ResponseMode.NO_TEXT,
            service_context=self.service_context,
        )
        return query_engine

    def _retrieve(
        self,
        query: str,
        indices: List[str] | None = None,
        language: str | None = None,
        top_k: int | None = None,
        include_tags: List[str] | None = None,
        include_web_results: bool | None = False,
        **kwargs,
    ):
        """Retrieves the top k results from the index for the given query.

        Args:
            query: A string representing the query.
            indices: A list of strings representing the indices to retrieve the results from.
            language: A string representing the language of the query.
            top_k: An integer representing the number of top results to retrieve.
            include_tags: A list of strings representing the tags to include in the results.
            exclude_tags: A list of strings representing the tags to exclude from the results.

        Returns:
            A list of dictionaries representing the retrieved results.
        """
        top_k = top_k or self.config.top_k
        language = language or self.config.language

        retriever = HybridRetriever(
            index=self.index,
            storage_context=self.storage_context,
            similarity_top_k=self.config.similarity_top_k,
            language=language,
            indices=indices,
            include_tags=include_tags,
            include_web_results=include_web_results,
        )

        retrieval_engine = self.load_query_engine(
            retriever=retriever,
            top_k=top_k,
            language=language,
        )

        query_bundle = QueryBundle(
            query_str=query,
            embedding=self.service_context.embed_model.get_query_embedding(
                query=query
            ),
        )
        results = retrieval_engine.retrieve(query_bundle)

        outputs = [
            {
                "text": node.get_text(),
                "metadata": node.metadata,
                "score": node.get_score(),
            }
            for node in results
        ]
        self.is_avoid_query = None
        return outputs

    def retrieve(
        self,
        query: str,
        language: str = "en",
        indices: List[str] | None = None,
        top_k: int = 5,
        include_tags: List[str] | None = None,
        include_web_results: bool | None = False,
        **kwargs,
    ):
        """Retrieves the top k results from the index for the given query.

        Args:
            query: A string representing the query.
            language: A string representing the language of the query.
            indices: A list of strings representing the indices to retrieve the results from.
            top_k: An integer representing the number of top results to retrieve.
            include_tags: A list of strings representing the tags to include in the results.
            include_web_results: A boolean representing whether to include web results.

        Returns:
            A list of dictionaries representing the retrieved results.
        """

        return self._retrieve(
            query,
            indices=indices if indices else [],
            language=language,
            top_k=top_k,
            include_tags=include_tags if include_tags else [],
            include_web_results=include_web_results,
        )

    def __call__(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        retrievals = self.retrieve(query, **kwargs)
        logger.debug(f"Retrieved {len(retrievals)} results.")
        logger.debug(f"Retrieval: {retrievals[0]}")
        return retrievals