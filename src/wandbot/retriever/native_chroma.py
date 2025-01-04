"""Native ChromaDB implementation that matches langchain-chroma behavior.

This module provides a native ChromaDB implementation that replaces the langchain-chroma
dependency while maintaining exact compatibility with its behavior, including:
- Same distance metrics and relevance scoring
- Identical MMR implementation using cosine similarity
- Matching query parameters and filtering
"""

import os
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import TypeAlias
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import chromadb
from chromadb.utils import embedding_functions as chromadb_ef

Matrix: TypeAlias = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance.
    
    Args:
        query_embedding: query embedding
        embedding_list: list of embeddings to consider
        lambda_mult: lambda parameter (0 for MMR, 1 for standard similarity)
        k: number of documents to return

    Returns:
        List of indices of selected embeddings
    """
    # Handle empty case
    if min(k, len(embedding_list)) <= 0:
        return []
    
    # Convert embeddings to numpy array
    embedding_list = np.array(embedding_list)
    
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # Calculate similarity to query
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    
    # First selection is most similar to query
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    
    # Iteratively select most diverse documents
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        
        # Calculate similarity to already selected docs
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        
        # Calculate MMR score for each remaining document
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            
            # Calculate diversity penalty
            redundant_score = max(similarity_to_selected[i])
            
            # Calculate MMR score
            equation_score = lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    
    return idxs


class NativeChromaWrapper:
    """Native ChromaDB wrapper that matches langchain-chroma's interface exactly.
    
    This class provides a drop-in replacement for langchain-chroma's Chroma class,
    implementing identical behavior including:
    - Same distance metrics and relevance scoring
    - Identical MMR implementation
    - Matching query parameters and filtering
    """
    
    def __init__(self, collection, embedding_function, override_relevance_score_fn: Optional[Callable] = None):
        """Initialize the wrapper.
        
        Args:
            collection: ChromaDB collection
            embedding_function: Function to generate embeddings
            override_relevance_score_fn: Optional function to override relevance scoring
        """
        self.collection = collection
        self.embedding_function = embedding_function
        self.override_relevance_score_fn = override_relevance_score_fn
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,  # Match langchain default
        filter: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results to return (default: 4)
            filter: Optional metadata filter
            where_document: Optional document content filter
            
        Returns:
            List of Documents
        """
        docs_and_scores = self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            where_document=where_document
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,  # Match langchain default
        filter: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with relevance scores.
        
        Args:
            query: Query text
            k: Number of results to return (default: 4)
            filter: Optional metadata filter
            where_document: Optional document content filter
            
        Returns:
            List of (Document, score) tuples, where score is the relevance score
        """
        # Get query results
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Get relevance score function based on distance metric
        relevance_score_fn = self._select_relevance_score_fn()
        
        # Convert to Documents with scores
        docs_and_scores = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Add relevance score to metadata
            meta = meta or {}
            score = relevance_score_fn(dist)
            meta["relevance_score"] = score
            
            # Get source content if available
            content = meta.get("source_content", doc)
            
            docs_and_scores.append(
                (Document(page_content=content, metadata=meta), score)
            )
        
        return docs_and_scores
    
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the appropriate relevance score function based on distance metric.
        
        The relevance function may differ depending on:
        - Distance/similarity metric used (cosine, l2, ip)
        - Scale of embeddings (unit normed vs not)
        - Embedding dimensionality
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        distance = "l2"  # Default to l2 distance
        distance_key = "hnsw:space"
        metadata = self.collection.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]

        if distance == "cosine":
            return lambda x: 1.0 - x  # Convert cosine distance to similarity
        elif distance == "l2":
            return lambda x: 1.0 / (1.0 + x)  # Convert L2 distance to similarity
        elif distance == "ip":
            return lambda x: x  # Inner product is already a similarity
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance metric of type: {distance}."
                "Consider providing relevance_score_fn to Chroma constructor."
            )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,  # Match langchain default
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform MMR search using exact langchain-chroma implementation.
        
        Args:
            query: Query text
            k: Number of results to return (default: 4)
            fetch_k: Number of initial candidates to fetch (default: 20)
            lambda_mult: MMR diversity weight (default: 0.5)
            filter: Optional metadata filter
            where_document: Optional document content filter
            
        Returns:
            List of Documents
        """
        # Match langchain-chroma behavior: fetch k*4 documents initially
        fetch_k = max(fetch_k, k * 4)
        
        # First get initial candidates
        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            where=filter,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )
        
        # Get query embedding
        query_embedding = self.embedding_function([query])[0]
        
        # Get document embeddings
        doc_embeddings = results['embeddings'][0]
        
        # Perform MMR reranking using langchain's implementation
        mmr_idxs = maximal_marginal_relevance(
            query_embedding=query_embedding,
            embedding_list=doc_embeddings,
            lambda_mult=lambda_mult,
            k=k
        )
        
        # Get relevance score function based on distance metric
        relevance_score_fn = self._select_relevance_score_fn()
        
        # Convert to Documents with metadata
        documents = []
        for idx in mmr_idxs:
            doc = results['documents'][0][idx]
            meta = results['metadatas'][0][idx] or {}
            dist = results['distances'][0][idx]
            
            # Add relevance score to metadata using appropriate scoring function
            meta["relevance_score"] = relevance_score_fn(dist)
            
            # Get source content if available
            content = meta.get("source_content", doc)
            
            documents.append(Document(page_content=content, metadata=meta))
        
        return documents
    
    def as_retriever(
        self,
        search_type: str = "mmr",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Return a retriever interface matching langchain-chroma exactly.
        
        Args:
            search_type: Type of search ("similarity", "mmr", or "similarity_score_threshold")
            search_kwargs: Search parameters
                k: Number of documents to return (default: 4)
                fetch_k: Number of documents to pass to MMR (default: k*4)
                lambda_mult: Diversity of results (default: 0.5)
                filter: Filter by metadata
                where_document: Filter by document content
                score_threshold: Minimum relevance score for similarity_score_threshold
            
        Returns:
            Retriever callable
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}  # Match langchain default
            
        def retrieve(query: str) -> List[Document]:
            k = search_kwargs.get("k", 4)
            filter_dict = search_kwargs.get("filter", None)
            where_document = search_kwargs.get("where_document", None)
            
            if search_type == "mmr":
                fetch_k = search_kwargs.get("fetch_k", k * 4)
                lambda_mult = search_kwargs.get("lambda_mult", 0.5)
                
                return self.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filter=filter_dict,
                    where_document=where_document
                )
            elif search_type == "similarity_score_threshold":
                score_threshold = search_kwargs.get("score_threshold", 0.0)
                results = self.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict,
                    where_document=where_document
                )
                return [doc for doc, score in results if score >= score_threshold]
            else:  # Default to similarity
                return self.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict,
                    where_document=where_document
                )
        
        return RunnableLambda(retrieve)