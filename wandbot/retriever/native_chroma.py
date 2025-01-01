"""Native ChromaDB implementation that uses chromadb's built-in distance metrics.

This module provides a native ChromaDB implementation that replaces the langchain-chroma
dependency. It uses chromadb's built-in distance metrics (cosine, l2, ip) for better
performance and compatibility.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import chromadb
from chromadb.utils import embedding_functions as chromadb_ef


class NativeChromaWrapper:
    """Native ChromaDB wrapper that matches langchain-chroma's interface.
    
    This class provides a drop-in replacement for langchain-chroma's Chroma class,
    implementing the same interface but using native chromadb operations for better
    performance.
    """
    
    def __init__(self, collection, embedding_function):
        """Initialize the wrapper.
        
        Args:
            collection: ChromaDB collection
            embedding_function: Function to generate embeddings
        """
        self.collection = collection
        self.embedding_function = embedding_function
    
    def similarity_search(
        self,
        query: str,
        k: int = 2,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Documents
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 2,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform MMR search using chromadb's built-in MMR.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of initial candidates to fetch
            lambda_mult: MMR diversity weight
            filter: Optional metadata filter
            
        Returns:
            List of Documents
        """
        # Use chromadb's built-in MMR
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter,
            include=['documents', 'metadatas'],
            query_type="mmr",
            mmr_lambda=lambda_mult,
            mmr_k=fetch_k
        )
        
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    
    def as_retriever(
        self,
        search_type: str = "mmr",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Return a retriever interface matching langchain-chroma.
        
        Args:
            search_type: Type of search ("similarity" or "mmr")
            search_kwargs: Search parameters
            
        Returns:
            Retriever callable
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        def retrieve(query: str) -> List[Document]:
            if search_type == "mmr":
                k = search_kwargs.get("k", 5)
                fetch_k = search_kwargs.get("fetch_k", min(k * 2, 20))
                lambda_mult = search_kwargs.get("lambda_mult", 0.5)
                filter_dict = search_kwargs.get("filter", None)
                
                return self.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filter=filter_dict
                )
            else:
                return self.similarity_search(
                    query=query,
                    **search_kwargs
                )
        
        return RunnableLambda(retrieve)


def setup_native_chroma(
    persist_dir: str,
    collection_name: str,
    embedding_model: str = "text-embedding-3-small",
    embedding_dimensions: int = 512,
    api_key: Optional[str] = None
) -> NativeChromaWrapper:
    """Setup a native chromadb vectorstore.
    
    Args:
        persist_dir: Directory to persist the database
        collection_name: Name of the collection
        embedding_model: OpenAI embedding model name
        embedding_dimensions: Embedding dimensions
        api_key: Optional OpenAI API key (defaults to env var)
        
    Returns:
        NativeChromaWrapper instance
    """
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Initialize OpenAI embeddings
    embedding_fn = chromadb_ef.OpenAIEmbeddingFunction(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model_name=embedding_model,
        api_base="https://api.openai.com/v1",
        model_kwargs={"dimensions": embedding_dimensions}
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    
    return NativeChromaWrapper(collection, embedding_fn)