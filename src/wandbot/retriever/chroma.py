"""Native ChromaDB implementation that matches langchain-chroma behavior.

This module provides a native ChromaDB implementation that replaces the langchain-chroma
dependency while maintaining exact compatibility with its behavior, including:
- Same distance metrics and relevance scoring
- Identical MMR implementation using cosine similarity
- Matching query parameters and filtering
"""
import asyncio
import weave
from typing import Any, Callable, Dict, List, Optional
import numpy as np

import chromadb
from chromadb.config import Settings
from wandbot.retriever.tmp_mmr import debug_max_marginal_relevance_search_by_vector

from wandbot.retriever.utils import maximal_marginal_relevance
from wandbot.utils import get_logger
from wandbot.models.embedding import EmbeddingModel
from wandbot.schema.document import Document

logger = get_logger(__name__)

class ChromaVectorStore:
    """Native ChromaDB wrapper that matches langchain-chroma's interface exactly.
    
    This class provides a drop-in replacement for langchain-chroma's Chroma class,
    implementing identical behavior including:
    - Same distance metrics and relevance scoring
    - Identical MMR implementation
    - Matching query parameters and filtering
    """
    
    def __init__(self, embedding_model, vector_store_config, chat_config, override_relevance_score_fn: Optional[Callable] = None):
        """Initialize the wrapper.
        
        Args:
            embedding_model: Embedding model    
            vector_store_config: Vector store config
            chat_config: Chat config
            override_relevance_score_fn: Optional function to override relevance scoring
        """
        self.embedding_model = embedding_model
        self.vector_store_config = vector_store_config
        self.chat_config = chat_config
        self.override_relevance_score_fn = override_relevance_score_fn
        self.chroma_vectorstore_client = chromadb.PersistentClient(
            path=str(self.vector_store_config.vectordb_index_dir),
            settings=Settings(anonymized_telemetry=False))
        self.collection = self.chroma_vectorstore_client.get_or_create_collection(
            name=self.vector_store_config.vectordb_collection_name,
            embedding_function=self.embedding_model,
        )
    
    @weave.op
    def query(self, 
              query_texts: Optional[List[str]] = None, 
              query_embeddings: Optional[List[float]] = None,
              n_results: int = 1, 
              filter: Optional[Dict[str, Any]] = None, 
              where_document: Optional[Dict[str, Any]] = None,
              include: List[str] = ['documents', 'metadatas', 'distances']
              ) -> Dict[str, List[Any]]:
        
        res = self.collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=filter,
            where_document=where_document,
            include=include
        )
        logger.debug(f"VECTORSTORE: {len(query_texts) if query_texts is not None else len(query_embeddings)} queries, {len(res)} vector store `.query` results \
of lengths: {[len(r) for r in res['documents']]}")
        return res
    
    @weave.op
    def embed_query(self, query_texts):
        return self.embedding_function(query_texts)

    @weave.op
    def similarity_search(
        self,
        query_texts:List[str],
        top_k: int,
        return_embeddings: bool = False,
        filter: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Document]]:
        """Perform similarity search.
        
        Args:
            query_texts: List of query texts
            top_k: Number of results to return
            return_embeddings: Whether to return embeddings
            filter: Optional metadata filter
            where_document: Optional document content filter
            
        Returns:
            Dict mapping query texts to lists of Documents, plus embedding status
        """
        
        return_components = ['documents', 'metadatas', 'distances']
        if return_embeddings:
            return_components.append("embeddings")

        query_embeddings, error_info = self.embedding_function.embed(query_texts)
        retrieved_results = self.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            filter=filter,
            where_document=where_document,
            include=return_components
        )

        logger.debug(f"SIMILARITY-SEARCH: {len(retrieved_results['documents'])} sets of results returned from similarity search call.")
        logger.debug(f"SIMILARITY-SEARCH: First query text:\n{query_texts[0]}\n")
        logger.debug(f"SIMILARITY-SEARCH: First returned documents:\n{retrieved_results['documents'][0]}\n")
        logger.debug(f"SIMILARITY-SEARCH: First returned document metadatas:\n{retrieved_results['metadatas'][0]}\n")
        logger.debug(f"SIMILARITY-SEARCH: First returned document distances:\n{retrieved_results['distances'][0]}\n")

        all_documents = []
        for i in range(len(retrieved_results['documents'])):
            docs = self._process_retrieved_results(
                retrieved_results['documents'][i], 
                retrieved_results['metadatas'][i], 
                retrieved_results['distances'][i]
            )
            all_documents.append(docs)

        # Link query to result, mostly for logging visibility
        results_dict = {}
        for i, result in enumerate(all_documents):
            results_dict[query_texts[i]] = result
            
        # Add embedding status to results
        results_dict["_embedding_status"] = error_info
        
        return results_dict

    @weave.op
    def max_marginal_relevance_search(
        self,
        query_texts: List[str],
        top_k: int,
        fetch_k: int,
        lambda_mult: float,
        filter: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Document]]:
        """
        Perform MMR search using exact implementation from retrieval_implementation.md.
        """
        # Get query embeddings using the embedding model
        query_embeddings, api_status = self.embedding_model.embed(query_texts)
        
        # Get initial results from ChromaDB with n_results=20
        logger.info(f"VECTORSTORE: Fetching {fetch_k} results from ChromaDB for MMR search")
        retrieved_results = self.query(
            query_embeddings=query_embeddings,
            n_results=fetch_k,  # Critical: This matches fetch_k in MMR
            filter=filter,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )
        
        # Log shapes and structure
        logger.debug("VECTORSTORE: Retrieved results structure:")
        logger.debug(f"- documents shape: {len(retrieved_results['documents'])} lists of {len(retrieved_results['documents'][0])} docs each")
        logger.debug(f"- embeddings shape: {len(retrieved_results['embeddings'])} lists of {len(retrieved_results['embeddings'][0])} embeddings each")
        logger.debug(f"- First document from first query: {retrieved_results['documents'][0][0][:100]}...")
        
        # Process each query sequentially since we can't use asyncio.gather with lists
        results = []
        for i, query_embed in enumerate(query_embeddings):
            # Structure results for this specific query
            query_results = {
                "documents": [retrieved_results["documents"][i]],  # Add extra dimension for this query's docs
                "metadatas": [retrieved_results["metadatas"][i]],
                "distances": [retrieved_results["distances"][i]],
                "embeddings": [retrieved_results["embeddings"][i]]
            }
            
            # Run MMR for this query
            mmr_results = debug_max_marginal_relevance_search_by_vector(
                results=query_results,
                embedding=query_embed,
                k=top_k,              # Final number of documents to return
                fetch_k=fetch_k,      # Must match n_results from ChromaDB query
                lambda_mult=lambda_mult  # Balance between relevance and diversity
            )
            results.append(mmr_results)
        
        # Link query to result and add embedding status
        results_dict = {}
        for i, result in enumerate(results):
            results_dict[query_texts[i]] = result
            
        results_dict["_embedding_status"] = api_status
        return results_dict

    def _process_retrieved_results(
        self,
        documents: List[str],
        metadatas: List[Dict],
        distances: List[float]
    ) -> List[Document]:
        """Convert retrieved results to Documents with scores."""
        relevance_score_fn = self._select_relevance_score_fn()
        processed_docs = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta or {}
            score = relevance_score_fn(dist)
            meta["relevance_score"] = score
            content = meta.get("source_content", doc)
            processed_docs.append(Document(page_content=content, metadata=meta, id=meta.get("id")))
        return processed_docs

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

    def _convert_to_numpy(self, embeddings):
        """Convert embeddings to numpy arrays if they aren't already"""
        if isinstance(embeddings, np.ndarray):
            return embeddings
        return np.array(embeddings)