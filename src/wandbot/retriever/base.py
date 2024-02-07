import wandb
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from wandbot.ingestion.config import VectorStoreConfig


def load_vector_store_from_config(config: VectorStoreConfig):
    embedding_fn = OpenAIEmbeddings(
        model=config.embeddings_model, dimensions=config.embedding_dim
    )

    base_vectorstore = Chroma(
        collection_name=config.name,
        embedding_function=embedding_fn,
        persist_directory=str(config.persist_dir),
    )
    return base_vectorstore


def load_vector_store_from_artifact(artifact_url: str):
    artifact = wandb.run.use_artifact(artifact_url)
    artifact_dir = artifact.download()
    config = VectorStoreConfig(persist_dir=artifact_dir)
    base_vectorstore = load_vector_store_from_config(config)
    return base_vectorstore


def load_retriever_with_options(
    base_vectorstore, search_type="mmr", search_kwargs={"k": 5}
):
    base_retriever = base_vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return base_retriever
