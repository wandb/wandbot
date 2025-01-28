from typing import List, Dict
import asyncio
import weave
from wandbot.retriever.chroma import ChromaVectorStore
from wandbot.schema.document import Document
import wandb

from wandbot.models.embedding import EmbeddingModel
from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.configs.chat_config import ChatConfig
from wandbot.utils import get_logger
logger = get_logger(__name__)

class VectorStore:
    """
    Sets up vector store and embedding model.
    """
    
    def __init__(self, vector_store_config: VectorStoreConfig, chat_config: ChatConfig):
        self.vector_store_config = vector_store_config
        self.chat_config = chat_config
        try:
            self.query_embedding_function = EmbeddingModel(
                provider = self.vector_store_config.embeddings_provider,
                model_name = self.vector_store_config.embeddings_model_name,
                dimensions = self.vector_store_config.embeddings_dimensions,
                input_type = self.vector_store_config.embeddings_query_input_type,
                encoding_format = self.vector_store_config.embeddings_encoding_format
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model:\n{str(e)}\n") from e
        
        try:
            self.chroma_vectorstore = ChromaVectorStore(
                embedding_function=self.query_embedding_function,
                vector_store_config=self.vector_store_config,
                chat_config=self.chat_config
            )   
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store:\n{str(e)}\n") from e

    @classmethod
    def from_config(cls, vector_store_config: VectorStoreConfig, chat_config: ChatConfig):
        if vector_store_config.vectordb_index_dir.exists():
            return cls(vector_store_config=vector_store_config, chat_config=chat_config)
        else:
            api = wandb.Api()
            art = api.artifact(vector_store_config.vectordb_index_artifact_url)  # Download vectordb index from W&B
            _ = art.download(vector_store_config.vectordb_index_dir)
            return cls(vector_store_config=vector_store_config, chat_config=chat_config)

    @weave.op
    def retrieve(self, query_texts: List[str], search_type: str = "mmr", search_kwargs: dict = None) -> Dict[str, List[Document]]:
        """`retrieve` method returns a list of documents per query in query_texts."""
        search_kwargs = search_kwargs or {}

        if search_type == "mmr":
            results = self.chroma_vectorstore.max_marginal_relevance_search(
                query_texts=query_texts,
                top_k=self.chat_config.top_k_per_query,
                fetch_k=self.chat_config.fetch_k,
                lambda_mult=self.chat_config.mmr_lambda_mult,
                filter=search_kwargs.get("filter"),
                where_document=search_kwargs.get("where_document")
            )
        else: 
            results = self.chroma_vectorstore.similarity_search(
                query_texts=query_texts,
                top_k=self.chat_config.top_k_per_query,
                filter=search_kwargs.get("filter"),
                where_document=search_kwargs.get("where_document")
            )

        return results
    
    async def _async_retrieve(self, query_texts: List[str], search_type: str = "mmr", search_kwargs: dict = None) -> List[Document]:
        return await asyncio.to_thread(
            self.retrieve,
            query_texts=query_texts,
            search_type=search_type,
            search_kwargs=search_kwargs,
        )