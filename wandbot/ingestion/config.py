"""Configuration for vector store."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store.
    
    Attributes:
        persist_dir: Directory to persist the database
        collection_name: Name of the collection
        embedding_model: OpenAI embedding model name
        embedding_dimensions: Embedding dimensions
        openai_api_key: OpenAI API key
    """

    persist_dir: Path = Field(
        Path("artifacts/wandbot_chroma_index:v0"),
        description="Directory to persist the database",
    )
    collection_name: str = Field(
        "vectorstore",
        description="Name of the collection",
    )
    embedding_model: str = Field(
        "text-embedding-3-small",
        description="OpenAI embedding model name",
    )
    embedding_dimensions: int = Field(
        512,
        description="Embedding dimensions",
    )
    openai_api_key: Optional[str] = Field(
        None,
        description="OpenAI API key (defaults to env var)",
    )