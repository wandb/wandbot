import datetime
import logging
import os
from typing import Any

import faiss
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from llama_index import (
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
)
from llama_index.llms import OpenAI
from llama_index.vector_stores import FaissVectorStore


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    )
    logger = logging.getLogger(name)
    return logger


class Timer:
    def __init__(self) -> None:
        self.start = datetime.datetime.utcnow()
        self.stop = self.start

    def __enter__(self) -> "Timer":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop = datetime.datetime.utcnow()

    @property
    def elapsed(self) -> float:
        return (self.stop - self.start).total_seconds()


def load_embeddings(cache_dir):
    underlying_embeddings = OpenAIEmbeddings()

    embeddings_cache_fs = LocalFileStore(cache_dir)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        embeddings_cache_fs,
        namespace=underlying_embeddings.model + "/",
    )
    return cached_embedder


def load_llm(model_name, temperature, max_retries):
    llm = OpenAI(model=model_name, temperature=temperature, streaming=True, max_retries=max_retries)
    return llm


def load_service_context(llm, temperature, embeddings_cache, max_retries, callback_manager=None):
    llm = load_llm(llm, temperature, max_retries=max_retries)
    embed_model = load_embeddings(embeddings_cache)
    return ServiceContext.from_defaults(llm=llm, embed_model=embed_model, callback_manager=callback_manager)


def load_storage_context(embed_dimensions, persist_dir):
    if os.path.isdir(persist_dir):
        storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_dir(persist_dir),
            persist_dir=persist_dir,
        )
    else:
        faiss_index = faiss.IndexFlatL2(embed_dimensions)
        storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore(faiss_index),
        )
    return storage_context


def load_index(nodes, service_context, storage_context, persist_dir):
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
