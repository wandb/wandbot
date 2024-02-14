import wandb
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from wandbot.ingestion.config import VectorStoreConfig
from wandbot.retriever.utils import OpenAIEmbeddingsModel


class VectorStore:
    embeddings_model: OpenAIEmbeddingsModel = OpenAIEmbeddingsModel(
        dimensions=512
    )

    def __init__(
        self, embeddings_model: str, collection_name: str, persist_dir: str
    ):
        self.embeddings_model = embeddings_model
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings_model,
            persist_directory=persist_dir,
        )

    @classmethod
    def from_config(cls, config: VectorStoreConfig):
        if config.persist_dir.exists():
            return cls(
                embeddings_model=config.embeddings_model,
                collection_name=config.name,
                persist_dir=str(config.persist_dir),
            )
        if wandb.run is None:
            api = wandb.Api()
            artifact = api.artifact(config.artifact_url)
        else:
            artifact = wandb.run.use_artifact(config.artifact_url)
        _ = artifact.download(root=str(config.persist_dir))

        return cls(
            embeddings_model=config.embeddings_model,
            collection_name=config.name,
            persist_dir=str(config.persist_dir),
        )

    def as_retriever(self, search_type="mmr", search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def as_parent_retriever(self, search_type="mmr", search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
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
