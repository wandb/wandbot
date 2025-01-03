"""Retriever package for wandbot.

This package provides the vector store and retrieval functionality for wandbot.
It includes a native ChromaDB implementation with optimized MMR search.
"""

from wandbot.retriever.base import VectorStore
from wandbot.retriever.native_chroma import NativeChromaWrapper, setup_native_chroma

__all__ = ["VectorStore", "NativeChromaWrapper", "setup_native_chroma"]