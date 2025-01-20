from typing import List
import asyncio
import weave
from wandbot.retriever.chroma import ChromaVectorStore
from langchain_core.documents import Document
import wandb

from wandbot.retriever.utils import EmbeddingsRedundantFilter
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
        if vector_store_config.index_dir.exists():
            return cls(vector_store_config=vector_store_config, chat_config=chat_config)
        else:
            api = wandb.Api()
            art = api.artifact(vector_store_config.artifact_url)  # Download vectordb index from W&B
            _ = art.download(vector_store_config.index_dir)
            return cls(vector_store_config=vector_store_config, chat_config=chat_config)

    @weave.op
    def retrieve(self, query_texts: List[str], search_kwargs: dict = None, return_scores: bool = False) -> List[Document]:
        """`retrieve` method returns a list of documents per query in query_texts."""
        search_kwargs = search_kwargs or {}

        if self.chat_config.search_type == "mmr":
            docs = self.chroma_vectorstore.max_marginal_relevance_search(
                query_texts=query_texts,
                top_k=self.chat_config.top_k_per_query,
                fetch_k=self.chat_config.fetch_k,
                lambda_mult=self.chat_config.mmr_lambda_mult,
                filter=search_kwargs.get("filter"),
                where_document=search_kwargs.get("where_document")
            )
        else: 
            docs = self.chroma_vectorstore.similarity_search(
                query_texts=query_texts,
                top_k=self.chat_config.top_k_per_query,
                filter=search_kwargs.get("filter"),
                where_document=search_kwargs.get("where_document")
            )

        # Handle flattening for (Document, score) tuples
        if isinstance(docs, list) and all(isinstance(d, list) for d in docs):
            flattened = [item for sublist in docs for item in sublist]  # flattens to List[Tuple[Document, float]]
            docs = flattened if return_scores else [doc for doc, _ in flattened]

        logger.info(f"VECTORSTORE: Retrieved {len(docs)} documents")
        logger.debug(f"VECTORSTORE: First retrieved document:\n{docs[0]}\n")
        return docs
    
    async def _async_retrieve(self, query_texts: List[str], search_kwargs: dict = None, return_scores: bool = False) -> List[Document]:
        return await asyncio.to_thread(
            self.retrieve,
            query_texts=query_texts,
            search_kwargs=search_kwargs,
            return_scores=return_scores
        )