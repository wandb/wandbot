"""Base VectorStore class for wandbot."""

from pathlib import Path
from typing import Optional

from wandbot.ingestion.config import VectorStoreConfig
from wandbot.retriever.native_chroma import NativeChromaWrapper
import chromadb
from chromadb.utils import embedding_functions as chromadb_ef


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

        # Initialize chromadb client
        client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Initialize OpenAI embeddings
        embedding_fn = chromadb_ef.OpenAIEmbeddingFunction(
            api_key=self.config.openai_api_key,
            model_name=self.config.embedding_model_name,
            api_base="https://api.openai.com/v1",
            model_kwargs={"dimensions": self.config.embedding_dimensions}
        )
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Create wrapper
        self.vectorstore = NativeChromaWrapper(collection, embedding_fn)

    def as_retriever(self, *args, **kwargs):
        """Return vectorstore as retriever.
        
        Args:
            *args: Positional arguments to pass to vectorstore
            **kwargs: Keyword arguments to pass to vectorstore
            
        Returns:
            Retriever interface
        """
        return self.vectorstore.as_retriever(*args, **kwargs)