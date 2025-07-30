import logging
import pathlib
from typing import List, Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class VectorStoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", 
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Vector Store
    vectordb_collection_name: str = "chroma_index-v56" #"vectorstore", vectorstore-chroma_index-v54
    vectordb_index_dir: pathlib.Path = Field(
        pathlib.Path("artifacts/vector_stores"), env="VECTORDB_INDEX_DIR"
    )
    vectordb_index_artifact_url: str = "wandbot/wandbot-dev/chroma_index:v56"
    distance: str = "l2"  # used in retrieval from vectordb 
    distance_key: str = "hnsw:space"  # used in retrieval from vectordb 
    
    # ChromaDB Client Mode
    vector_store_mode: Literal["local", "hosted"] = "hosted"
    # Settings for hosted mode (using direct HttpClient parameters)
    vector_store_host: Optional[str] = "api.trychroma.com"  # e.g., 'api.trychroma.com'
    vector_store_tenant: Optional[str] = '3c66fbfc-98ce-41ff-92ec-ef16e71c8c0a' # Tenant ID for hosted Chroma
    vector_store_database: Optional[str] = 'wandbot-prod' # Database name for hosted Chroma
    vector_store_api_key: Optional[str] = None # Pulled as env variable from .env file
    
    # Embeddings settings
    embeddings_provider:str = "openai"
    embeddings_model_name: str = "text-embedding-3-small"
    embeddings_dimensions: int = 512  # needed when using OpenAI embeddings
    embeddings_n_parallel_api_calls: int = 40
    # Embedding input types, e.g. "search_query" or "search_document"
    embeddings_query_input_type: str = "search_query"  # needed when using Cohere embeddings
    embeddings_document_input_type: str = "search_document"  # needed when using Cohere embeddings
    
    # Embedding encoding format
    embeddings_encoding_format: Literal["float", "base64"] = "base64"
    
    # Ingestion settings
    batch_size: int = 256
    persist_directory: Optional[pathlib.Path] = None

    # Remote ChromaDB Upload Configuration - Keys for transformation
    remote_chroma_keys_to_prepend: List[str] = Field(
        default_factory=lambda: [
            "parent_id",
            "id",
            "source",
            "file_type",
            "has_code",
            "source_type",
            "tags",
            "description",
        ]
    )
    remote_chroma_keys_to_remove: List[str] = Field(
        default_factory=lambda: [
            "source_content",
            "file_type",
            "tags",
            "description",
        ]
    )

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