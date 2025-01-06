from operator import itemgetter
from typing import List, Sequence, Any

import weave
from wandbot.retriever.chroma import ChromaWrapper
# from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel

import wandb
import chromadb
import chromadb.utils.embedding_functions as chroma_embedding_functions
from chromadb import Documents as ChromaDocuments
from chromadb import Embeddings as ChromaEmbeddings

from wandbot.configs.vectorstore_config import VectorStoreConfig
from wandbot.retriever.reranking import CohereRerankChain
from wandbot.retriever.utils import EmbeddingsRedundantFilter

from wandbot.configs.vectorstore_config import VectorStoreConfig
from wandbot.models.embedding import EmbeddingModel
from wandbot.retriever.utils import cosine_similarity
from wandbot.configs.chat_config import ChatConfig

class VectorStore:
    
    def __init__(self, vector_store_config: VectorStoreConfig, chat_config: ChatConfig):
        self.vector_store_config = vector_store_config
        self.chat_config = chat_config
        try:
            self.document_embedding_function = EmbeddingModel(
                provider = self.vector_store_config.embeddings_provider,
                model_name = self.vector_store_config.embeddings_model_name,
                dimensions = self.vector_store_config.embeddings_dimensions,
                input_type = self.vector_store_config.embeddings_document_input_type
            )
            self.query_embedding_function = EmbeddingModel(
                provider = self.vector_store_config.embeddings_provider,
                model_name = self.vector_store_config.embeddings_model_name,
                dimensions = self.vector_store_config.embeddings_dimensions,
                input_type = self.vector_store_config.embeddings_query_input_type
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding models: {str(e)}") from e
        
        try:
            self.vectorstore_client = chromadb.PersistentClient(path=str(self.vector_store_config.persist_dir))
            self.vectorstore_collection = self.vectorstore_client.get_or_create_collection(
                name=self.vector_store_config.collection_name,
                embedding_function=self.document_embedding_function,
            )
            self.vectorstore = ChromaWrapper(
                collection=self.vectorstore_collection,
                embedding_function=self.query_embedding_function,
                vector_store_config=self.vector_store_config,
                chat_config=self.chat_config
            )   
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}") from e

    @classmethod
    def from_config(cls, vector_store_config: VectorStoreConfig, chat_config: ChatConfig):
        if vector_store_config.persist_dir.exists():
            return cls(vector_store_config=vector_store_config, chat_config=chat_config)
        api = wandb.Api()
        _ = api.artifact(vector_store_config.artifact_url)  # Download vectordb index from W&B
        return cls(vector_store_config=vector_store_config, chat_config=chat_config)

    def as_retriever(self, search_kwargs: dict, search_type:str ="mmr"):
        return self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def as_parent_retriever(self, search_kwargs: dict, search_type:str ="mmr"):
        retriever = self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )
        parent_retriever = retriever | RunnableLambda(
            lambda docs: [
                Document(
                    page_content=doc.metadata.get(
                        "source_content", doc.page_content
                    ),
                    metadata=doc.metadata,
                )
                for doc in docs
            ]
        )
        return parent_retriever


class SimpleRetrievalEngine:
    cohere_rerank_chain = CohereRerankChain()

    def __init__(self, vector_store: VectorStore, chat_config: ChatConfig, rerank_models: dict):
        self.vector_store = vector_store
        self.chat_config = chat_config
        self.cohere_rerank_chain = rerank_models  # type: ignore
        self.redundant_filter = EmbeddingsRedundantFilter(
            embedding_function = self.vector_store.document_embedding_function,
            similarity_fn = cosine_similarity,
            redundant_similarity_threshold = self.chat_config.redundant_similarity_threshold,
        ).transform_documents

    @weave.op
    def retrieve_and_rerank(self, retriever, question: str, language: str | None = "en"):
        """
        Retrieve relevant documents and rerank them.
        
        Args:
            question: The query string
            language: Language filter for documents
            top_k: Number of documents to return
        """
        # Get initial docs from retriever
        retrieved_docs = retriever(question)
        
        # Filter out very similar documents
        filtered_docs = self.redundant_filter(retrieved_docs)
        
        # Apply reranking
        reranked_results = self.cohere_rerank_chain({
            "question": question,
            "language": language,
            "top_k": self.chat_config.top_k,
            "context": filtered_docs
        })
        return reranked_results

    @weave.op
    def __call__(
        self,
        question: str,
        top_k: int = None,
        language: str | None = "en",
        search_type: str = "mmr",
        sources: List[str] = None,
    ):
        if top_k is None:
            top_k = self.chat_config.top_k

        if search_type is None:
            search_type = self.chat_config.search_type

        filters = {}
        source_filter = None
        language_filter = None

        # Filter by sources
        if sources is not None:
            source_filter = {"source_type": {"$in": sources}}
        
        # Filter by language
        if language is not None:
            language_filter = {"language": language}
        
        if source_filter and language_filter:
            filters = {"$and": [source_filter, language_filter]}
        elif source_filter:
            filters = source_filter
        elif language_filter:
            filters = language_filter
        
        if filters:
            search_kwargs = {"filter": filters}
        else:
            search_kwargs = {}

        retriever = self.vector_store.as_parent_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        reranked_results = self.retrieve_and_rerank(question, retriever, language)

        # Format output
        outputs = []
        for result in reranked_results:
            result_dict = {
                "text": result.page_content,
                "score": result.metadata["relevance_score"],
            }
            metadata_dict = {
                k: v
                for k, v in result.metadata.items()
                if k not in ["relevance_score", "source_content", "id", "parent_id"]
            }
            result_dict["metadata"] = metadata_dict
            outputs.append(result_dict)
        
        return outputs