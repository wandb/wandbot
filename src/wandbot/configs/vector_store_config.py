import pathlib
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

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
    
    # Embeddings settings
    embeddings_provider:str = "openai"
    embeddings_model_name: str = "text-embedding-3-small"
    embeddings_dimensions: int = 512  # needed when using OpenAI embeddings
    
    # Embedding input types, e.g. "search_query" or "search_document"
    embeddings_query_input_type: str = "search_query"  # needed when using Cohere embeddings
    embeddings_document_input_type: str = "search_document"  # needed when using Cohere embeddings
    
    # Embedding encoding format
    embeddings_encoding_format: Literal["float", "base64"] = "base64"
    
    # Ingestion settings
    batch_size: int = 256  # used during ingestion when adding docs to vectorstore