from operator import itemgetter
from typing import List

import weave
from langchain_chroma import Chroma
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel

import wandb
from wandbot.ingestion.config import VectorStoreConfig
from wandbot.retriever.reranking import CohereRerankChain
from wandbot.retriever.utils import OpenAIEmbeddingsModel


class VectorStore:
    embeddings_model: OpenAIEmbeddingsModel = OpenAIEmbeddingsModel()
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._vectorstore = None  # Lazy initialization
        self.embeddings_model = {
            "embedding_model_name": self.config.embedding_model_name,
            "tokenizer_model_name": self.config.tokenizer_model_name,
            "embedding_dimensions": self.config.embedding_dimensions,
        }

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                embedding_function=self.embeddings_model,
                collection_name=self.config.collection_name,
                persist_directory=str(self.config.persist_dir),
            )
        return self._vectorstore

    @classmethod
    def from_config(cls, config: VectorStoreConfig):
        if config.persist_dir.exists():
            return cls(config=config)
        if wandb.run is None:
            api = wandb.Api()
            artifact = api.artifact(config.artifact_url)
        else:
            artifact = wandb.run.use_artifact(config.artifact_url)
        _ = artifact.download(root=str(config.persist_dir))
        return cls(config=config)

    @classmethod
    async def initialize(cls, config: VectorStoreConfig):
        """Async initialization method"""
        instance = cls(config=config)
        if not config.persist_dir.exists():
            if wandb.run is None:
                api = wandb.Api()
                artifact = api.artifact(config.artifact_url)
            else:
                artifact = wandb.run.use_artifact(config.artifact_url)
            await wandb.run.loop.run_in_executor(
                None, artifact.download, str(config.persist_dir)
            )
        return instance
        
    def as_retriever(self, search_type="mmr", search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vectorstore.as_retriever(search_type=search_type,
                                             search_kwargs=search_kwargs)

    def as_parent_retriever(self, search_type="mmr", search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        retriever = self.vectorstore.as_retriever(search_type=search_type,
                                                  search_kwargs=search_kwargs)
        parent_retriever = retriever | RunnableLambda(lambda docs: [
            Document(
                page_content=doc.metadata.get("source_content", doc.
                                              page_content),
                metadata=doc.metadata,
            ) for doc in docs
        ])
        return parent_retriever


class SimpleRetrievalEngine:
    top_k: int = 5
    cohere_rerank_chain = CohereRerankChain()

    def __init__(self, vector_store: VectorStore, rerank_models: dict):
        self.vector_store = vector_store
        self.cohere_rerank_chain = rerank_models  # type: ignore
        self.embeddings_model = self.vector_store.embeddings_model
        self.redundant_filter = EmbeddingsRedundantFilter(
            embeddings=self.embeddings_model).transform_documents

    @weave.op()
    def __call__(
        self,
        question: str,
        language: str | None = None,
        top_k: int = 5,
        search_type="mmr",
        sources: List[str] = None,
    ):
        filters = {}
        source_filter = None
        language_filter = None
        if sources is not None:
            source_filter = {"source_type": {"$in": sources}}
        if language is not None:
            language_filter = {"language": language}
        if source_filter and language_filter:
            filters = {"$and": [source_filter, language_filter]}
        elif source_filter:
            filters = source_filter
        elif language_filter:
            filters = language_filter
        if filters:
            search_kwargs = {"k": top_k * 4, "filter": filters}
        else:
            search_kwargs = {"k": top_k * 4}

        retriever = self.vector_store.as_parent_retriever(
            search_type=search_type, search_kwargs=search_kwargs)

        retrieval_chain = (RunnableParallel(
            question=itemgetter("question"),
            language=itemgetter("language"),
            context=(itemgetter("question") | retriever
                     | self.redundant_filter),
        )
                           | self.cohere_rerank_chain)
        results = retrieval_chain.invoke({
            "question": question,
            "language": language,
            "top_k": top_k
        })
        outputs = []
        for result in results:
            result_dict = {
                "text": result.page_content,
                "score": result.metadata["relevance_score"],
            }
            metadata_dict = {
                k: v
                for k, v in result.metadata.items() if k not in
                ["relevance_score", "source_content", "id", "parent_id"]
            }
            result_dict["metadata"] = metadata_dict
            outputs.append(result_dict)

        return outputs