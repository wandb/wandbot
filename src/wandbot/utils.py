"""This module contains utility functions and classes for the Wandbot system.

The module includes the following functions:
- `get_logger`: Creates and returns a logger with the specified name.
- `load_embeddings`: Loads embeddings from cache or creates new ones if not found.
- `load_llm`: Loads a language model with the specified parameters.
- `load_service_context`: Loads a service context with the specified parameters.
- `load_storage_context`: Loads a storage context with the specified parameters.
- `load_index`: Loads an index from storage or creates a new one if not found.

The module also includes the following classes:
- `Timer`: A simple timer class for measuring elapsed time.

Typical usage example:

    logger = get_logger("my_logger")
    embeddings = load_embeddings("/path/to/cache")
    llm = load_llm("gpt-3", 0.5, 3)
    service_context = load_service_context(llm, 0.5, "/path/to/cache", 3)
    storage_context = load_storage_context(768, "/path/to/persist")
    index = load_index(nodes, service_context, storage_context, "/path/to/persist")
"""
import datetime
import logging
import os
from typing import Any, List, Optional

import faiss
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.bridge.pydantic import Field
from llama_index.llms import OpenAI
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores import FaissVectorStore


def get_logger(name: str) -> logging.Logger:
    """Creates and returns a logger with the specified name.

    Args:
        name: The name of the logger.

    Returns:
        A logger instance with the specified name.
    """
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    )
    logger = logging.getLogger(name)
    return logger


class Timer:
    """A simple timer class for measuring elapsed time."""

    def __init__(self) -> None:
        """Initializes the timer."""
        self.start = datetime.datetime.utcnow()
        self.stop = self.start

    def __enter__(self) -> "Timer":
        """Starts the timer."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Stops the timer."""
        self.stop = datetime.datetime.utcnow()

    @property
    def elapsed(self) -> float:
        """Calculates the elapsed time in seconds."""
        return (self.stop - self.start).total_seconds()


def load_embeddings(cache_dir: str) -> CacheBackedEmbeddings:
    """Loads embeddings from cache or creates new ones if not found.

    Args:
        cache_dir: The directory where the embeddings cache is stored.

    Returns:
        A cached embedder instance.
    """
    underlying_embeddings = OpenAIEmbeddings()

    embeddings_cache_fs = LocalFileStore(cache_dir)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        embeddings_cache_fs,
        namespace=underlying_embeddings.model + "/",
    )
    return cached_embedder


def load_llm(model_name: str, temperature: float, max_retries: int) -> OpenAI:
    """Loads a language model with the specified parameters.

    Args:
        model_name: The name of the model to load.
        temperature: The temperature parameter for the model.
        max_retries: The maximum number of retries for loading the model.

    Returns:
        An instance of the loaded language model.
    """
    llm = OpenAI(
        model=model_name,
        temperature=temperature,
        streaming=True,
        max_retries=max_retries,
    )
    return llm


def load_service_context(
    llm: str,
    temperature: float,
    embeddings_cache: str,
    max_retries: int,
    callback_manager: Optional[Any] = None,
) -> ServiceContext:
    """Loads a service context with the specified parameters.

    Args:
        llm: The language model to load.
        temperature: The temperature parameter for the model.
        embeddings_cache: The directory where the embeddings cache is stored.
        max_retries: The maximum number of retries for loading the model.
        callback_manager: The callback manager for the service context (optional).

    Returns:
        A service context instance with the specified parameters.
    """
    llm = load_llm(llm, temperature, max_retries=max_retries)
    embed_model = load_embeddings(embeddings_cache)
    return ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, callback_manager=callback_manager
    )


def load_storage_context(embed_dimensions: int) -> StorageContext:
    """Loads a storage context with the specified parameters.

    Args:
        embed_dimensions: The dimensions of the embeddings.

    Returns:
        A storage context instance with the specified parameters.
    """

    faiss_index = faiss.IndexFlatL2(embed_dimensions)
    storage_context = StorageContext.from_defaults(
        vector_store=FaissVectorStore(faiss_index),
    )
    return storage_context


def load_index(
    nodes: Any,
    service_context: ServiceContext,
    storage_context: StorageContext,
    persist_dir: str,
) -> VectorStoreIndex:
    """Loads an index from storage or creates a new one if not found.

    Args:
        nodes: The nodes to include in the index.
        service_context: The service context for the index.
        storage_context: The storage context for the index.
        persist_dir: The directory where the index is persisted.

    Returns:
        An index instance with the specified parameters.
    """
    try:
        index = load_index_from_storage(storage_context)
    except Exception:
        index = VectorStoreIndex(
            nodes=nodes,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=persist_dir)
    return index


class LanguageFilterPostprocessor(BaseNodePostprocessor):
    """Language-based Node processor."""

    languages: List[str] = Field(default=["en", "python"])
    min_result_size: int = Field(default=10)

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
