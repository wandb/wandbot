"""Base VectorStore class for wandbot."""

from pathlib import Path
from typing import Optional

from wandbot.ingestion.config import VectorStoreConfig
from wandbot.retriever.native_chroma import setup_native_chroma


class VectorStore:
    """Base VectorStore class that handles initialization and configuration."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize VectorStore.
        
        Args:
            config: VectorStore configuration
        """
        self.config = config
        self.vectorstore = None

    @classmethod
    def from_config(cls, config: VectorStoreConfig):
        """Create VectorStore from config.
        
        Args:
            config: VectorStore configuration
            
        Returns:
            VectorStore instance
        """
        instance = cls(config)
        instance._initialize()
        return instance

    def _initialize(self):
        """Initialize the vectorstore."""
        # Create persist directory if it doesn't exist
        persist_dir = Path(self.config.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Setup native chromadb
        self.vectorstore = setup_native_chroma(
            persist_dir=str(persist_dir),
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model,
            embedding_dimensions=self.config.embedding_dimensions,
            api_key=self.config.openai_api_key
        )

    def as_retriever(self, *args, **kwargs):
        """Return vectorstore as retriever.
        
        Args:
            *args: Positional arguments to pass to vectorstore
            **kwargs: Keyword arguments to pass to vectorstore
            
        Returns:
            Retriever interface
        """
        return self.vectorstore.as_retriever(*args, **kwargs)