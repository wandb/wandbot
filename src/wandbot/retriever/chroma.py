"""Native ChromaDB implementation that matches langchain-chroma behavior.

This module provides a native ChromaDB implementation that replaces the langchain-chroma
dependency while maintaining exact compatibility with its behavior, including:
- Same distance metrics and relevance scoring
- Identical MMR implementation using cosine similarity
- Matching query parameters and filtering
"""
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

import chromadb
import numpy as np
import weave
from chromadb.config import Settings
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from wandbot.configs.chat_config import ChatConfig
from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.models.embedding import EmbeddingModel
from wandbot.retriever.mmr import max_marginal_relevance_search_by_vector
from wandbot.schema.document import Document
from wandbot.utils import get_logger

logger = get_logger(__name__)
retry_chat_config = ChatConfig()

# Helper retry decorator configuration
vector_store_retry_decorator = retry(
    retry=retry_if_exception_type(Exception),  # Retry on any exception for broad coverage
    stop=stop_after_attempt(retry_chat_config.vector_store_max_retries),
    wait=wait_exponential(
        multiplier=retry_chat_config.vector_store_retry_multiplier,
        min=retry_chat_config.vector_store_retry_min_wait,
        max=retry_chat_config.vector_store_retry_max_wait
    ),
    before_sleep=lambda retry_state: (
        before_sleep_log(logger, log_level=logging.WARNING)(retry_state),
        logger.warning(f"Retrying vector store operation due to {retry_state.outcome.exception()}. Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep:.2f} seconds...")
    )
)

class ChromaVectorStore:
    """Native ChromaDB wrapper that matches langchain-chroma's interface exactly.
    
    This class provides a drop-in replacement for langchain-chroma's Chroma class,
    implementing identical behavior including:
    - Same distance metrics and relevance scoring
    - Identical MMR implementation
    - Matching query parameters and filtering
    """
    def __init__(self, embedding_model: EmbeddingModel, vector_store_config: VectorStoreConfig, chat_config: ChatConfig, override_relevance_score_fn: Optional[Callable] = None):
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

        # Initialize ChromaDB client based on mode
        if self.vector_store_config.vector_store_mode == "local":
            logger.info(f"Initializing vector store in local mode from: {self.vector_store_config.vectordb_index_dir}")
            self.chroma_vectorstore_client = chromadb.PersistentClient(
                path=str(self.vector_store_config.vectordb_index_dir),
                settings=Settings(anonymized_telemetry=False)
            )
        elif self.vector_store_config.vector_store_mode == "hosted":
            host = self.vector_store_config.vector_store_host
            tenant = self.vector_store_config.vector_store_tenant
            database = self.vector_store_config.vector_store_database
            api_key = self.vector_store_config.vector_store_api_key

            logger.info(f"Initializing vector store in hosted mode: host={host}, tenant={tenant}, database={database}")

            if not host:
                 raise ValueError("vector_store_host must be set in VectorStoreConfig for hosted mode")
            if not tenant:
                 raise ValueError("vector_store_tenant must be set in VectorStoreConfig for hosted mode")
            if not database:
                 raise ValueError("vector_store_database must be set in VectorStoreConfig for hosted mode")

            headers = {}
            if api_key:
                logger.info("Adding x-chroma-token header for authentication.")
                headers['x-chroma-token'] = api_key
            else:
                # Depending on the hosted provider, lack of token might be an error
                logger.warning("No vector_store_api_key found in config for hosted mode. Connecting without authentication header.")

            # Note: Removed Settings object approach, using direct parameters now
            self.chroma_vectorstore_client = chromadb.HttpClient(
                host=host,
                ssl=True,
                tenant=tenant,
                database=database,
                headers=headers,
                settings=Settings(anonymized_telemetry=False) # Keep basic settings if needed
            )
        else:
            raise ValueError(f"Invalid vector_store_mode: {self.vector_store_config.vector_store_mode}")
            
        logger.info(f"Initializing Chroma collection: {self.vector_store_config.vectordb_collection_name}")
        # Prepare metadata to be passed during creation/retrieval
        collection_metadata = {self.vector_store_config.distance_key: self.vector_store_config.distance}

        logger.info(f"ChromaVectorStore: Attempting to get or create collection '{self.vector_store_config.vectordb_collection_name}'...")
        self.collection = self.chroma_vectorstore_client.get_or_create_collection(
            name=self.vector_store_config.vectordb_collection_name,
            embedding_function=self.embedding_model,
            metadata=collection_metadata  # Pass metadata here
        )
        logger.info(f"ChromaVectorStore: Successfully got or created collection '{self.collection.name}'.")
        
        # Verify metadata after creation/retrieval (optional logging)
        retrieved_metadata = self.collection.metadata or {}
        if self.vector_store_config.distance_key not in retrieved_metadata or \
           retrieved_metadata.get(self.vector_store_config.distance_key) != self.vector_store_config.distance:
             logger.warning(f"Retrieved collection metadata {retrieved_metadata} does not match expected {collection_metadata}. "
                            f"This might happen if the collection existed with different settings. Relevance scoring might be affected.")
        else:
            logger.info(f"Successfully confirmed collection metadata: {retrieved_metadata}")

    @weave.op
    @vector_store_retry_decorator
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
        return self.embedding_model(query_texts)

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
        if not api_status.success:
            # Safely access error message, provide default if unavailable
            error_message = "Unknown embedding error"
            if api_status.error_info and hasattr(api_status.error_info, 'error_message'):
                error_message = api_status.error_info.error_message
            raise RuntimeError(f"Embedding failed: {error_message}")
            
        # Get initial results from ChromaDB
        logger.info(f"VECTORSTORE: Fetching {len(query_texts) * fetch_k} results from ChromaDB for MMR search")
        retrieved_results = self.query(
            query_embeddings=query_embeddings,
            n_results=fetch_k,
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
            retrieved_results_dict = {
                "documents": retrieved_results["documents"][i],  # Add extra dimension for this query's docs
                "metadatas": retrieved_results["metadatas"][i],
                "distances": retrieved_results["distances"][i],
                "embeddings": retrieved_results["embeddings"][i]
            }
            
            # Run MMR for this query
            mmr_results = max_marginal_relevance_search_by_vector(
                retrieved_results=retrieved_results_dict,
                embedding=query_embed,
                top_k=top_k,              # Final number of documents to return
                lambda_mult=lambda_mult  # Balance between relevance and diversity
            )
            results.append(mmr_results)
        
        # Link query to result and add embedding status
        results_dict = {}
        for i, result in enumerate(results):
            results_dict[query_texts[i]] = result
            
        results_dict["_embedding_status"] = api_status
        return results_dict

    @weave.op
    def add_documents(
        self, 
        documents: List[Document], 
        ids: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> List[str]:
        """Run documents through the embeddings and add them to the vectorstore.

        Args:
            documents (List[Document]): Documents to add to the vectorstore.
            ids (Optional[List[str]], optional): Optional list of IDs. Defaults to None.
            kwargs (Any): Additional keyword arguments (currently ignored).

        Returns:
            List[str]: List of IDs of the added documents.
            
        Raises:
            ValueError: If the number of provided IDs does not match the number of documents.
            RuntimeError: If the embedding process fails.
        """
        texts_to_embed = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Handle IDs: Use provided, then doc.id, then generate UUIDs
        if ids is None:
            # final_ids = [doc.id if doc.id else str(uuid.uuid4()) for doc in documents] # Incorrect: Document has no id attribute
            # Always generate UUIDs if no IDs are provided
            final_ids = [str(uuid.uuid4()) for _ in documents]
        else:
            if len(ids) != len(documents):
                raise ValueError("Number of IDs provided does not match number of documents")

        # Embed the documents using the stored embedding model instance
        logger.info(f"Generating embeddings for {len(texts_to_embed)} documents...")
        embeddings, api_status = self.embedding_model.embed(texts_to_embed)
        
        if not api_status.success:
            logger.error(f"Embedding failed: {api_status.error_info.error_message}")
            # Depending on desired behavior, either raise or return partial/empty results
            raise RuntimeError(f"Embedding failed: {api_status.error_info.error_message}")
            # return [] # Option: return empty list on failure

        logger.info(f"Adding {len(final_ids)} documents to Chroma collection '{self.collection.name}'...")
        # Use the native chromadb client's add method
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts_to_embed, # Native client expects text content here
            ids=final_ids
        )
        logger.info("Documents added successfully.")
        return final_ids

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
            meta["relevance_score"] = meta.get("relevance_score", score) 
            content = meta.get("source_content", doc)
            doc_id = meta.get("id", None) 
            processed_docs.append(Document(page_content=content, metadata=meta, id=doc_id))
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
        else:
             logger.warning(f"Distance key '{self.vector_store_config.distance_key}' not found in collection metadata. Defaulting to '{distance}'.")

        if distance == "cosine":
            space = metadata.get("hnsw:space", "l2").lower() # Default to l2 if not present
            logger.debug(f"Selecting relevance score function based on distance '{distance}' and space '{space}'")
            if space == "cosine": # Cosine Similarity
                 return lambda x: x
            elif space == "ip": # Inner Product
                 return lambda x: x
            else: # Defaulting to L2 or if space is explicitly L2 but distance is 'cosine' (confusing)
                 # This matches Langchain's formula for cosine distance -> similarity
                 return lambda x: 1.0 - x  
        elif distance == "l2":
            # Matches Langchain's L2 normalization
            return lambda x: 1.0 / (1.0 + x**2) # Squaring L2 distance is common
        elif distance == "ip":
             # Inner product is already a similarity measure
            return lambda x: x
        else:
            logger.error(f"Unsupported distance metric '{distance}'. Defaulting to no normalization.")
            # Fallback or raise error - returning raw distance if unsure
            raise ValueError(
                "No supported normalization function"
                f" for distance metric of type: {distance}."
                "Consider providing relevance_score_fn to Chroma constructor."
            )

    def _convert_to_numpy(self, embeddings):
        """Convert embeddings to numpy arrays if they aren't already"""
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        elif embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings