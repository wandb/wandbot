"""Native ChromaDB implementation with optimized MMR search.

This module provides a native ChromaDB implementation that replaces the langchain-chroma
dependency. It includes an optimized Maximum Marginal Relevance (MMR) implementation
adapted from langchain-chroma's implementation.

Key optimizations:
1. Pre-computing all similarities at once
2. Using numpy's vectorized operations for masking and selection
3. Avoiding redundant computations by reusing pre-computed similarities
4. Using memory views for efficient array operations

Credit: The MMR implementation is adapted from langchain-chroma's implementation:
https://github.com/langchain-ai/langchain/blob/master/libs/langchain-chroma/langchain_chroma/vectorstores.py
"""

import os
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import chromadb
from chromadb.utils import embedding_functions as chromadb_ef


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute cosine similarity using memory views for efficiency.
    
    Args:
        X: First array of shape (M, D)
        Y: Second array of shape (N, D)
        
    Returns:
        Similarity matrix of shape (M, N)
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    # Ensure contiguous arrays
    X_cont = np.ascontiguousarray(X, dtype=np.float64)
    Y_cont = np.ascontiguousarray(Y, dtype=np.float64)
    
    # Create memory views
    X_view = memoryview(X_cont)
    Y_view = memoryview(Y_cont)
    
    # Convert back to numpy arrays for computation
    X_arr = np.frombuffer(X_view, dtype=np.float64).reshape(X.shape)
    Y_arr = np.frombuffer(Y_view, dtype=np.float64).reshape(Y.shape)
    
    X_norm = np.linalg.norm(X_arr, axis=1)
    Y_norm = np.linalg.norm(Y_arr, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X_arr, Y_arr.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def compute_mmr(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    lambda_mult: float = 0.5,
    k: int = 2
) -> List[Document]:
    """Compute Maximum Marginal Relevance (MMR).
    
    This implementation is adapted from langchain-chroma's maximal_marginal_relevance function:
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain-chroma/langchain_chroma/vectorstores.py
    
    Key optimizations:
    1. Pre-computing all similarities at once
    2. Using numpy's vectorized operations for masking and selection
    3. Avoiding redundant computations by reusing pre-computed similarities
    
    Args:
        query_embedding: Query embedding of shape (D,)
        embeddings: Document embeddings of shape (N, D)
        documents: List of document texts
        metadatas: List of document metadata
        lambda_mult: MMR diversity weight (0 = max diversity, 1 = max relevance)
        k: Number of documents to return
        
    Returns:
        List of selected Documents
    """
    if min(k, len(documents)) <= 0:
        return []
    
    # Ensure query_embedding is 2D
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # Pre-compute all similarities at once
    similarity_to_query = cosine_similarity(query_embedding, embeddings)[0]
    
    # Select first document (most similar to query)
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    
    # Pre-compute all pairwise similarities
    all_similarities = cosine_similarity(embeddings, embeddings)
    
    # Select remaining documents
    while len(idxs) < min(k, len(documents)):
        # Get similarities to selected documents for unselected documents
        unselected_mask = ~np.isin(range(len(embeddings)), idxs)
        sim_to_selected = all_similarities[unselected_mask][:, idxs]
        max_sim_to_selected = np.max(sim_to_selected, axis=1)
        
        # Calculate MMR scores for unselected documents
        unselected_scores = similarity_to_query[unselected_mask]
        mmr_scores = lambda_mult * unselected_scores - (1 - lambda_mult) * max_sim_to_selected
        
        # Select document with highest MMR score
        idx_to_add = np.where(unselected_mask)[0][np.argmax(mmr_scores)]
        idxs.append(idx_to_add)
    
    return [
        Document(page_content=documents[idx], metadata=metadatas[idx])
        for idx in idxs
    ]


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
        """Perform MMR search.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of initial candidates to fetch
            lambda_mult: MMR diversity weight
            filter: Optional metadata filter
            
        Returns:
            List of Documents
        """
        # Get query embedding
        query_embedding = self.embedding_function([query])[0]
        
        # Get more results than needed for MMR
        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            where=filter,
            include=['documents', 'metadatas', 'embeddings']
        )
        
        return compute_mmr(
            query_embedding=np.array(query_embedding),
            embeddings=np.array(results['embeddings'][0]),
            documents=results['documents'][0],
            metadatas=results['metadatas'][0],
            lambda_mult=lambda_mult,
            k=k
        )
    
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