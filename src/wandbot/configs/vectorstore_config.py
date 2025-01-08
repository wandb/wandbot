import pathlib
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class VectorStoreConfig(BaseSettings):
    # Vector Store
    collection_name: str = "vectorstore"
    index_dir: pathlib.Path = pathlib.Path("data/cache/vectorstore")
    artifact_url: str = "wandbot/wandbot-dev/chroma_index:v31"
    distance: str = "l2"  # used in retrieval from vectordb 
    distance_key: str = "hnsw:space"  # used in retrieval from vectordb 
    # Embeddings settings
    embeddings_provider:str = "openai"
    embeddings_model_name: str = "text-embedding-3-small"
    embeddings_dimensions: int = 512  # needed when using OpenAI embeddings
    embeddings_query_input_type: str = "search_query"  # needed when using Cohere embeddings
    embeddings_document_input_type: str = "search_document"  # needed when using Cohere embeddings
    # Ingestions
    batch_size: int = 256  # used during ingestion when adding docs to vectorstore