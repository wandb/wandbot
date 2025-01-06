"""Native ChromaDB implementation that matches langchain-chroma behavior.

This module provides a native ChromaDB implementation that replaces the langchain-chroma
dependency while maintaining exact compatibility with its behavior, including:
- Same distance metrics and relevance scoring
- Identical MMR implementation using cosine similarity
- Matching query parameters and filtering
"""
import weave
from typing import Any, Callable, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import numpy as np

from wandbot.retriever.utils import maximal_marginal_relevance


class ChromaWrapper:
    """Native ChromaDB wrapper that matches langchain-chroma's interface exactly.
    
    This class provides a drop-in replacement for langchain-chroma's Chroma class,
    implementing identical behavior including:
    - Same distance metrics and relevance scoring
    - Identical MMR implementation
    - Matching query parameters and filtering
    """
    
    def __init__(self, collection, embedding_function, vector_store_config, chat_config, override_relevance_score_fn: Optional[Callable] = None):
        """Initialize the wrapper.
        
        Args:
            collection: ChromaDB collection
            embedding_function: Function to generate embeddings
            override_relevance_score_fn: Optional function to override relevance scoring
        """
        self.collection = collection
        self.embedding_function = embedding_function
        self.vector_store_config = vector_store_config
        self.chat_config = chat_config
        self.override_relevance_score_fn = override_relevance_score_fn
    
    @weave.op
    def similarity_search(
        self,
        query: str,
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
            k=self.chat_config.top_k,
            filter=filter,
            where_document=where_document
        )
        return [doc for doc, _ in docs_and_scores]

    @weave.op
    def similarity_search_with_score(
        self,
        query: str,
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
            n_results=self.chat_config.top_k,
            where=filter,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )
        
        # Convert embeddings to numpy for MMR if needed
        if 'embeddings' in results:
            results['embeddings'] = [
                self._convert_to_numpy(emb) for emb in results['embeddings']
            ]
        
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

        distance = self.vector_store_config.distance  # Default
        metadata = self.collection.metadata

        if metadata and self.vector_store_config.distance_key in metadata:
            distance = metadata[self.vector_store_config.distance_key]

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

    @weave.op
    def max_marginal_relevance_search(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform MMR search using exact langchain-chroma implementation.
        
        Args:
            query: Query text
            filter: Optional metadata filter
            where_document: Optional document content filter
            
        Returns:
            List of Documents
        """
        
        # First get initial candidates
        results = self.collection.query(
            query_texts=[query],
            n_results=self.chat_config.fetch_k,
            where=filter,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )
        
        query_embedding = self._convert_to_numpy(self.embedding_function([query])[0])
        doc_embeddings = results['embeddings'][0]
        
        # Perform MMR reranking using langchain's implementation
        mmr_idxs = maximal_marginal_relevance(
            query_embedding=query_embedding,
            embedding_list=doc_embeddings,
            lambda_mult=self.chat_config.mmr_lambda_mult,
            top_k=self.chat_config.top_k
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
        search_kwargs: dict,
        search_type: str = "mmr",
    ):
        """Return a retriever interface matching langchain-chroma exactly.
        
        Args:
            search_type: Type of search ("similarity", "mmr", or "similarity_score_threshold")
            search_kwargs: Search parameters
                filter: Filter by metadata
                where_document: Filter by document content
                score_threshold: Minimum relevance score for similarity_score_threshold
            
        Returns:
            Retriever callable
        """

        @weave.op
        def retrieve(query: str) -> List[Document]:
            filter_dict = search_kwargs.get("filter", None)
            where_document = search_kwargs.get("where_document", None)
            
            if search_type == "mmr":
                return self.max_marginal_relevance_search(
                    query=query,
                    filter=filter_dict,
                    where_document=where_document
                )
            elif search_type == "similarity_score_threshold":
                similarity_score_threshold = self.chat_config.similarity_score_threshold
                results = self.similarity_search_with_score(
                    query=query,
                    filter=filter_dict,
                    where_document=where_document
                )
                return [doc for doc, score in results if score >= similarity_score_threshold]
            else:  # Default to similarity
                return self.similarity_search(
                    query=query,
                    filter=filter_dict,
                    where_document=where_document
                )
        
        return RunnableLambda(retrieve)

    def _convert_to_numpy(self, embeddings):
        """Convert embeddings to numpy arrays if they aren't already"""
        if isinstance(embeddings, np.ndarray):
            return embeddings
        return np.array(embeddings)