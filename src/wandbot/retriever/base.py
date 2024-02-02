import os
from typing import Any, Dict, List, Optional, Tuple

import wandb
from llama_index import (
    QueryBundle,
    ServiceContext,
    StorageContext,
    load_indices_from_storage,
)
from llama_index.callbacks import CallbackManager
from llama_index.core.base_retriever import BaseRetriever
from llama_index.postprocessor import BaseNodePostprocessor, CohereRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import BaseSynthesizer, ResponseMode
from llama_index.retrievers.fusion_retriever import FUSION_MODES
from llama_index.schema import NodeWithScore
from llama_index.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.vector_stores.types import DEFAULT_PERSIST_FNAME
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from wandbot.retriever.external import YouRetriever
from wandbot.retriever.fusion import FusionRetriever, HybridRetriever
from wandbot.retriever.postprocessors import (
    LanguageFilterPostprocessor,
    MetadataPostprocessor,
)
from wandbot.utils import get_logger, load_service_context, load_storage_context

logger = get_logger(__name__)


class WandbRetrieverQueryEngine(RetrieverQueryEngine):
    def __init__(
        self,
        retriever: FusionRetriever,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager,
        )
        self._retriever = retriever

    def retrieve(
        self, query_bundle: QueryBundle, **kwargs
    ) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle, **kwargs)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)


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

        (
            self.storage_context,
            index_ids,
        ) = self.load_storage_context_from_artifact(
            artifact_url=self.config.index_artifact
        )

        self.indices = load_indices_from_storage(
            self.storage_context,
            service_context=self.service_context,
            index_ids=index_ids,
        )
        retriever_list = []
        for index in self.indices:
            retriever = HybridRetriever(
                index=index,
                similarity_top_k=self.config.similarity_top_k,
                storage_context=self.storage_context,
            )
            retriever_list.append(retriever)
        self.you_retriever = YouRetriever(
            api_key=os.environ.get("YOU_API_KEY"),
            similarity_top_k=self.config.similarity_top_k,
        )
        self._retriever = FusionRetriever(
            retriever_list,
            similarity_top_k=self.config.similarity_top_k
            * len(retriever_list)
            // 2,
            num_queries=1,
            use_async=True,
            mode=FUSION_MODES.RECIPROCAL_RANK,
        )
        retriever_list.append(self.you_retriever)
        index_ids = index_ids + ["you.com"]
        self._retriever_map = dict(zip(index_ids, retriever_list))

        self.is_avoid_query: bool | None = None

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
        storage_context = load_storage_context(
            embed_dimensions=self.config.embeddings_size,
            persist_dir=artifact_dir,
        )
        return storage_context, artifact.metadata["index_ids"]

    def load_query_engine(
        self,
        retriever: BaseRetriever | None = None,
        top_k: int | None = None,
        language: str | None = None,
        include_tags: List[str] | None = None,
        exclude_tags: List[str] | None = None,
        is_avoid_query: bool | None = None,
    ) -> WandbRetrieverQueryEngine:
        top_k = top_k or self.config.top_k
        language = language or self.config.language

        if is_avoid_query is not None:
            self.is_avoid_query = is_avoid_query

        node_postprocessors = [
            MetadataPostprocessor(
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                min_result_size=top_k,
            ),
            LanguageFilterPostprocessor(
                languages=[language, "python"], min_result_size=top_k
            ),
            CohereRerank(top_n=top_k, model="rerank-english-v2.0")
            if language == "en"
            else CohereRerank(top_n=top_k, model="rerank-multilingual-v2.0"),
        ]
        query_engine = WandbRetrieverQueryEngine.from_args(
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
        exclude_tags: List[str] | None = None,
        is_avoid_query: bool | None = False,
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
        retrievers = []
        for index in indices or []:
            retriever = self._retriever_map.get(index)
            if not retriever and index:
                logger.warning(
                    f"Index {index} not found in retriever map. Skipping the index"
                )
            retrievers.append(retriever)

        retrievers = [retriever for retriever in retrievers if retriever]

        if not retrievers:
            logger.warning("No retrievers found. Defaulting to all retrievers")
            retriever = self._retriever
        else:
            retriever = FusionRetriever(
                retrievers,
                similarity_top_k=self.config.similarity_top_k
                * len(retrievers)
                // 2,
                num_queries=1,
                use_async=True,
                mode=FUSION_MODES.RECIPROCAL_RANK,
            )

        retrieval_engine = self.load_query_engine(
            retriever=retriever,
            top_k=top_k,
            language=language,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

        avoid_query = self.is_avoid_query or is_avoid_query

        query_bundle = QueryBundle(
            query_str=query,
            embedding=self.service_context.embed_model.get_query_embedding(
                query=query
            ),
        )
        results = retrieval_engine.retrieve(
            query_bundle, is_avoid_query=bool(avoid_query)
        )

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
        language: str | None = None,
        top_k: int | None = None,
        include_tags: List[str] | None = None,
        exclude_tags: List[str] | None = None,
        is_avoid_query: bool | None = False,
        **kwargs,
    ):
        """Retrieves the top k results from the index for the given query.

        Args:
            query: A string representing the query.
            language: A string representing the language of the query.
            top_k: An integer representing the number of top results to retrieve.
            include_tags: A list of strings representing the tags to include in the results.
            exclude_tags: A list of strings representing the tags to exclude from the results.

        Returns:
            A list of dictionaries representing the retrieved results.
        """

        return self._retrieve(
            query,
            indices=None,
            language=language,
            top_k=top_k,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            is_avoid_query=is_avoid_query,
        )

    def retrieve_from_indices(
        self,
        query: str,
        indices: List[str],
        language: str | None = None,
        top_k: int | None = None,
        include_tags: List[str] | None = None,
        exclude_tags: List[str] | None = None,
        is_avoid_query: bool | None = False,
    ):
        """Retrieves the top k results from the index for the given query.

        Args:
            query: A string representing the query.
            indices: A string representing the index to retrieve the results from.
            language: A string representing the language of the query.
            top_k: An integer representing the number of top results to retrieve.
            include_tags: A list of strings representing the tags to include in the results.
            exclude_tags: A list of strings representing the tags to exclude from the results.

        Returns:
            A list of dictionaries representing the retrieved results.
        """
        return self._retrieve(
            query,
            indices=indices,
            language=language,
            top_k=top_k,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            is_avoid_query=is_avoid_query,
        )

    def __call__(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        indices = kwargs.pop("indices", [])
        if kwargs.get("include_web_results"):
            indices.append("you.com")
        if len(indices) > 1:
            retrievals = self.retrieve_from_indices(
                query, indices=indices, **kwargs
            )
        else:
            retrievals = self.retrieve(query, **kwargs)

        logger.debug(f"Retrieved {len(retrievals)} results.")
        logger.debug(f"Retrieval: {retrievals[0]}")
        return retrievals
