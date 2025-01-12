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
from langchain_core.documents import Document
import numpy as np

import chromadb
from chromadb.config import Settings

from wandbot.retriever.utils import maximal_marginal_relevance
from wandbot.utils import get_logger

logger = get_logger(__name__)

class ChromaVectorStore:
    """Native ChromaDB wrapper that matches langchain-chroma's interface exactly.
    
    This class provides a drop-in replacement for langchain-chroma's Chroma class,
    implementing identical behavior including:
    - Same distance metrics and relevance scoring
    - Identical MMR implementation
    - Matching query parameters and filtering
    """
    
    def __init__(self, embedding_function, vector_store_config, chat_config, override_relevance_score_fn: Optional[Callable] = None):
        """Initialize the wrapper.
        
        Args:
            collection: ChromaDB collection
            embedding_function: Function to generate embeddings
            override_relevance_score_fn: Optional function to override relevance scoring
        """
        self.embedding_function = embedding_function
        self.vector_store_config = vector_store_config
        self.chat_config = chat_config
        self.override_relevance_score_fn = override_relevance_score_fn
        self.chroma_vectorstore_client = chromadb.PersistentClient(
            path=str(self.vector_store_config.index_dir),
            settings=Settings(anonymized_telemetry=False))
        self.collection = self.chroma_vectorstore_client.get_or_create_collection(
            name=self.vector_store_config.collection_name,
            embedding_function=self.embedding_function,
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
        logger.debug(f"VECTORSTORE: {len(res)} vector store `.query` results.")
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
    ) -> List[List[tuple[Document, float]]]:
        """Perform similarity search.
        
        Args:
            query_texts: List of query texts
            top_k: Number of results to return
            return_embeddings: Whether to return embeddings
            filter: Optional metadata filter
            where_document: Optional document content filter
            
        Returns:
            List of lists of (Document, score)
        """
        
        return_components = ['documents', 'metadatas', 'distances']
        if return_embeddings:
            return_components.append("embeddings")

        query_embeddings =self.embed_query(query_texts)
        retrieved_results = self.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            filter=filter,
            where_document=where_document,
            include=return_components
        )

        logger.debug(f"{len(retrieved_results['documents'])} sets of results returned from similarity search call.")

        all_documents = []
        for i in range(len(retrieved_results['documents'])):
            docs = self._process_retrieved_results(
                retrieved_results['documents'][i], 
                retrieved_results['metadatas'][i], 
                retrieved_results['distances'][i]
            )
            all_documents.append(docs)
        
        return all_documents

    @weave.op
    def max_marginal_relevance_search(
        self,
        query_texts: List[str],
        top_k: int,
        fetch_k: int,
        lambda_mult: float,
        filter: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[List[tuple[Document, float]]]:
        """
        Perform MMR search using exact langchain-chroma implementation.
        
        Returns:
            List of lists of (Document, score)
        """
        
        query_embeddings = self.embedding_function(query_texts)
        retrieved_results = self.query(
            query_embeddings=query_embeddings,
            n_results=fetch_k,
            filter=filter,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )
        logger.debug(f"VECTORSTORE:{len(retrieved_results['documents'])} sets of results returned from MMR search call.")
    
        async def run_mmr_batch(
            query_embedding: List[float],
            doc_embeddings: List[float],
            docs: List[str],
            metadatas: List[Dict[str, Any]],
            distances: List[float],
            top_k: int,
            lambda_mult: float
        ):
            # Perform MMR reranking using langchain's implementation
            mmr_idxs = maximal_marginal_relevance(
                query_embedding=self._convert_to_numpy(query_embedding),
                embedding_list=doc_embeddings,
                lambda_mult=lambda_mult,
                top_k=top_k
            )

            # return a list of Documents with metadata and scores
            return self._process_retrieved_results(
                *zip(*[(docs[i], metadatas[i] or {}, distances[i]) for i in mmr_idxs])
            )

        # Get or create event loop without closing it
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        tasks = [run_mmr_batch(
            query_embed,
            doc_embed,
            docs,
            metadatas,
            distances,
            top_k,
            lambda_mult
        ) for query_embed, doc_embed, docs, metadatas, distances in zip(
            query_embeddings,
            retrieved_results["embeddings"],
            retrieved_results["documents"],
            retrieved_results["metadatas"],
            retrieved_results["distances"]
        )]
        
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        return results

    def _process_retrieved_results(
        self,
        documents: List[str],
        metadatas: List[Dict],
        distances: List[float]
    ) -> List[tuple[Document, float]]:
        """Convert retrieved results to Documents with scores."""
        relevance_score_fn = self._select_relevance_score_fn()
        processed_docs = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta or {}
            score = relevance_score_fn(dist)
            meta["relevance_score"] = score
            content = meta.get("source_content", doc)
            processed_docs.append((Document(page_content=content, metadata=meta), score))
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