import pathlib
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from pydantic import model_validator
import logging

logger = logging.getLogger(__name__)

class VectorStoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", 
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Vector Store
    vectordb_collection_name: str = "vectorstore"
    vectordb_index_dir: pathlib.Path = pathlib.Path("data/cache/vectorstore")
    vectordb_index_artifact_url: str = "wandbot/wandbot-dev/chroma_index:v43"
    distance: str = "l2"  # used in retrieval from vectordb 
    distance_key: str = "hnsw:space"  # used in retrieval from vectordb 
    
    # ChromaDB Client Mode
    vector_store_mode: Literal["local", "hosted"] = "local"
    vector_store_host: Optional[str] = None  
    vector_store_port: Optional[int] = None  
    vector_store_auth_token: Optional[str] = None # pulled from .env
    
    # Embeddings settings
    embeddings_provider:str = "openai"
    embeddings_model_name: str = "text-embedding-3-large"
    embeddings_dimensions: int = 3072  # needed when using OpenAI embeddings
    
    # Embedding input types, e.g. "search_query" or "search_document"
    embeddings_query_input_type: str = "search_query"  # needed when using Cohere embeddings
    embeddings_document_input_type: str = "search_document"  # needed when using Cohere embeddings
    
    # Embedding encoding format
    embeddings_encoding_format: Literal["float", "base64"] = "base64"
    
    # Ingestion settings
    batch_size: int = 256  # used during ingestion when adding docs to vectorstore

    @model_validator(mode="after")
    def _adjust_paths_for_dimension(cls, values: "VectorStoreConfig") -> "VectorStoreConfig":
        """Adjusts index directory path based on embedding dimension."""
        if values.vector_store_mode == "local": # Only adjust for local mode
            base_dir = values.vectordb_index_dir
            dimension = values.embeddings_dimensions
            # Ensure we don't append dimension multiple times if already present
            if not base_dir.name.endswith(f"_{dimension}"):
                values.vectordb_index_dir = base_dir.parent / f"{base_dir.name}_{dimension}"
                logger.info(f"Adjusted vectordb_index_dir for dimension {dimension}: {values.vectordb_index_dir}")
        return values