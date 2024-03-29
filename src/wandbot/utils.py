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
import hashlib
import json
import logging
import os
import pathlib
import sqlite3
from typing import Any, List, Optional

import faiss
import fasttext
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import LiteLLM
from llama_index.llms.llm import LLM
from llama_index.schema import NodeWithScore, TextNode
from llama_index.vector_stores import FaissVectorStore
from pydantic_settings import BaseSettings

import wandb


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
        self.start = datetime.datetime.now().astimezone(datetime.timezone.utc)
        self.stop = self.start

    def __enter__(self) -> "Timer":
        """Starts the timer."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Stops the timer."""
        self.stop = datetime.datetime.now().astimezone(datetime.timezone.utc)

    @property
    def elapsed(self) -> float:
        """Calculates the elapsed time in seconds."""
        return (self.stop - self.start).total_seconds()


def load_embeddings(cache_dir: str) -> OpenAIEmbedding:
    """Loads embeddings from cache or creates new ones if not found.

    Args:
        cache_dir: The directory where the embeddings cache is stored.

    Returns:
        A cached embedder instance.
    """
    # underlying_embeddings = OpenAIEmbeddings()
    #
    # embeddings_cache_fs = LocalFileStore(cache_dir)
    # cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    #     underlying_embeddings,
    #     embeddings_cache_fs,
    #     namespace=underlying_embeddings.model + "/",
    # )
    #
    # return cast(LCEmbeddings, cached_embedder)
    embeddings = OpenAIEmbedding()
    return embeddings


def load_llm(
    model_name: str,
    temperature: float,
    max_retries: int,
) -> LLM:
    """Loads a language model with the specified parameters.

    Args:
        model_name: The name of the model to load.
        temperature: The temperature parameter for the model.
        max_retries: The maximum number of retries for loading the model.

    Returns:
        An instance of the loaded language model.
    """
    import litellm
    from litellm.caching import Cache

    litellm.cache = Cache()

    llm = LiteLLM(
        model=model_name,
        temperature=temperature,
        max_retries=max_retries,
        caching=True,
    )

    return llm


def load_service_context(
    embeddings_cache: str,
    llm: str = "gpt-3.5-turbo-16k-0613",
    temperature: float = 0.1,
    max_retries: int = 2,
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

    embed_model = load_embeddings(embeddings_cache)
    llm = load_llm(
        model_name=llm,
        temperature=temperature,
        max_retries=max_retries,
    )

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


def cachew(cache_path: str = "./cache.db", logger=None):
    """
    Memoization decorator that caches the output of a method in a SQLite
    database.
    ref: https://www.kevinkatz.io/posts/memoize-to-sqlite
    """
    db_conn = sqlite3.connect(cache_path)
    db_conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, result TEXT)"
    )

    def memoize(func):
        def wrapped(*args, **kwargs):
            # Compute the hash of the <function name>:<argument>
            xs = f"{func.__name__}:{repr(tuple(args))}:{repr(kwargs)}".encode(
                "utf-8"
            )
            arg_hash = hashlib.sha256(xs).hexdigest()

            # Check if the result is already cached
            cursor = db_conn.cursor()
            cursor.execute(
                "SELECT result FROM cache WHERE hash = ?", (arg_hash,)
            )
            row = cursor.fetchone()
            if row is not None:
                if logger is not None:
                    logger.debug(
                        f"Cached result found for {arg_hash}. Returning it."
                    )
                return json.loads(row[0])

            # Compute the result and cache it
            result = func(*args, **kwargs)
            cursor.execute(
                "INSERT INTO cache (hash, result) VALUES (?, ?)",
                (arg_hash, json.dumps(result)),
            )
            db_conn.commit()

            return result

        return wrapped

    return memoize


def create_no_result_dummy_node() -> NodeWithScore:
    """
    Creates a dummy node to be used when no results found.
    This can be used instead of returning results that are not relevant
    or have already been filtered out.
    """
    dummy_text = "No results found"
    dummy_metadata = {
        "source": "no-result",
        "language": "en",
        "description": "This is a dummy node when there are no results",
        "title": "No Result Node",
        "tags": ["no-result"],
    }
    dummy_text_node = TextNode(text=dummy_text, metadata=dummy_metadata)
    return NodeWithScore(node=dummy_text_node, score=0.0)


class FasttextModelConfig(BaseSettings):
    fasttext_file_path: pathlib.Path = pathlib.Path(
        "data/cache/models/lid.176.bin"
    )
    fasttext_artifact_name: str = (
        "wandbot/wandbot_public/fasttext-lid.176.bin:v0"
    )
    fasttext_artifact_type: str = "fasttext-model"


class FastTextLangDetect:
    """Uses fasttext to detect the language of a text, from this file:
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    """

    def __init__(self, config: FasttextModelConfig = FasttextModelConfig()):
        self.config = config
        self._model = self._load_model()

    def detect_language(self, text: str):
        predictions = self.model.predict(text.replace("\n", " "))
        return predictions[0][0].replace("__label__", "")

    def detect_language_batch(self, texts: List[str]):
        predictions = self.model.predict(texts)
        return [p[0].replace("__label__", "") for p in predictions[0]]

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        if not os.path.isfile(self.config.fasttext_file_path):
            _ = wandb.run.use_artifact(
                self.config.fasttext_artifact_name,
                type=self.config.fasttext_artifact_type,
            ).download(root=str(self.config.fasttext_file_path.parent))
        self._model = fasttext.load_model(str(self.config.fasttext_file_path))
        return self._model
