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

import asyncio
import datetime
import hashlib
import json
import logging
import os
import yaml
import pathlib
import re
import sqlite3
import string

import fasttext
import nest_asyncio
import tiktoken
from langchain_core.documents import Document
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Coroutine, List, Tuple, Optional

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
        level=logging.getLevelName(os.environ.get("LOG_LEVEL", "CRITICAL")),
    )
    logger = logging.getLogger(name)
    return logger


logger = get_logger(__name__)


def strip_punctuation(text):
    # Create a translation table mapping every punctuation character to None
    translator = str.maketrans("", "", string.punctuation)

    # Use the table to strip punctuation from the text
    no_punct = text.translate(translator)
    return no_punct


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


class FasttextModelConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    fasttext_file_path: pathlib.Path = pathlib.Path(
        "data/cache/models/lid.176.bin"
    )
    fasttext_artifact_path: str = Field(
        "wandbot/wandbot_public/fasttext-lid.176.bin:v0",
        env="LANGDETECT_ARTIFACT_PATH",
        validation_alias="langdetect_artifact_path",
    )
    fasttext_artifact_type: str = "fasttext-model"
    wandb_project: str = Field(
        "wandbot-dev", env="WANDB_PROJECT", validation_alias="wandb_project"
    )
    wandb_entity: str = Field(
        "wandbot", env="WANDB_ENTITY", validation_alias="wandb_entity"
    )


class FastTextLangDetect:
    """Uses fasttext to detect the language of a text, from this file:
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    """

    def __init__(self, config: FasttextModelConfig = FasttextModelConfig()):
        self.config = config
        self._model = self._load_model()

    def detect_language(self, text: str):
        cleaned_text = strip_punctuation(text).replace("\n", " ")
        predictions = self.model.predict(cleaned_text)
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
            if wandb.run is None:
                api = wandb.Api()
                artifact = api.artifact(self.config.fasttext_artifact_path)
            else:
                artifact = wandb.run.use_artifact(
                    self.config.fasttext_artifact_path,
                    type=self.config.fasttext_artifact_type,
                )
            _ = artifact.download(
                root=str(self.config.fasttext_file_path.parent)
            )
        self._model = fasttext.load_model(str(self.config.fasttext_file_path))
        return self._model


def run_async_tasks(
    tasks: List[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> Tuple[Any]:
    """Run a list of async tasks."""
    tasks_to_execute: List[Any] = tasks

    nest_asyncio.apply()
    if show_progress:
        try:
            from tqdm.asyncio import tqdm

            # jupyter notebooks already have an event loop running
            # we need to reuse it instead of creating a new one

            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> List[Any]:
                return await tqdm.gather(
                    *tasks_to_execute, desc=progress_bar_desc
                )

            tqdm_outputs: Tuple[Any] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        except Exception:
            pass

    async def _gather() -> Tuple[Any]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: Tuple[Any] = asyncio.run(_gather())
    return outputs


def clean_document_content(doc: Document) -> Document:
    cleaned_content = re.sub(r"\n{3,}", "\n\n", doc.page_content)
    cleaned_content = cleaned_content.strip()
    cleaned_document = Document(
        page_content=cleaned_content, metadata=doc.metadata
    )
    cleaned_document = make_document_tokenization_safe(cleaned_document)
    return cleaned_document


def make_document_tokenization_safe(document: Document) -> Document:
    """Removes special tokens from the given documents.

    Args:
        documents: A list of strings representing the documents.

    Returns:
        A list of cleaned documents with special tokens removed.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    special_tokens_set = encoding.special_tokens_set

    def remove_special_tokens(text: str) -> str:
        """Removes special tokens from the given text.

        Args:
            text: A string representing the text.

        Returns:
            The text with special tokens removed.
        """
        for token in special_tokens_set:
            text = text.replace(token, "")
        return text

    content = document.page_content
    cleaned_document = remove_special_tokens(content)
    return Document(page_content=cleaned_document, metadata=document.metadata)


def filter_smaller_documents(
    documents: List[Document], min_size: int = 3, min_line_size: int = 5
) -> List[Document]:
    def filter_small_document(document: Document) -> bool:
        return (
            len(
                [
                    doc
                    for doc in document.page_content.split("\n")
                    if len(doc.strip().split()) >= min_line_size
                ]
            )
            >= min_size
        )

    return [
        document for document in documents if filter_small_document(document)
    ]


class LLMConfig(BaseModel):
    model: str
    temperature: float
    max_retries: int


class FeatureToggle(BaseModel):
    enabled: bool


class EmbeddingsConfig(BaseModel):
    type: str
    config: dict


class RAGPipelineConfig(BaseModel):
    llm: LLMConfig
    fallback_llm: LLMConfig
    embeddings: EmbeddingsConfig
    retrieval_re_ranker: FeatureToggle
    use_you_search_api: FeatureToggle
    query_enhancer_followed_by_rerank_fusion: FeatureToggle
    chunk_size: int
    top_k: int
    project: str | None = Field("wandbot_public", env="WANDB_PROJECT")
    entity: str | None = Field("wandbot", env="WANDB_ENTITY")


def load_config(config_path: str) -> RAGPipelineConfig:
    """Load and return the YAML configuration as a Pydantic model."""
    config_path = pathlib.Path(__file__).parent / config_path
    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, 'r') as file:
        raw_config = yaml.safe_load(file)
    return RAGPipelineConfig(**raw_config)
