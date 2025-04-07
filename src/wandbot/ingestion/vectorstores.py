"""This module contains functions for loading and managing vector stores in the Wandbot ingestion system.

The module includes the following functions:
- `load`: Loads the vector store from the specified source artifact path and returns the name of the resulting artifact.

Typical usage example:

    project = "wandbot-dev"
    entity = "wandbot"
    source_artifact_path = "wandbot/wandbot-dev/raw_dataset:latest"
    result_artifact_name = "wandbot_index"
    load(project, entity, source_artifact_path, result_artifact_name)
"""

import json
import pathlib
from typing import List

# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from tqdm import trange

import wandb
from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.configs.chat_config import ChatConfig
from wandbot.models.embedding import EmbeddingModel
from wandbot.utils import get_logger
from wandbot.schema.document import Document
from wandbot.retriever.chroma import ChromaVectorStore

logger = get_logger(__name__)


def load(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = "wandbot_index",
) -> str:
    """Load the vector store.

    Loads the vector store from the specified source artifact path and returns the name of the resulting artifact.

    Args:
        project: The name of the project.
        entity: The name of the entity.
        source_artifact_path: The path to the source artifact.
        result_artifact_name: The name of the resulting artifact. Defaults to "wandbot_index".

    Returns:
        The name of the resulting artifact.

    Raises:
        wandb.Error: An error occurred during the loading process.
    """
    logger.info(f"Starting vector store creation for {entity}/{project} from artifact {source_artifact_path}")
    config: VectorStoreConfig = VectorStoreConfig()
    chat_config: ChatConfig = ChatConfig()
    run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
        project=project, entity=entity, job_type="create_vectorstore"
    )
    logger.info(f"Using source artifact: {source_artifact_path}")
    artifact: wandb.Artifact = run.use_artifact(
        source_artifact_path, type="dataset"
    )
    logger.info("Downloading source artifact...")
    artifact_dir: str = artifact.download()
    logger.info(f"Source artifact downloaded to: {artifact_dir}")

    logger.info(
        f"Initializing embedding model: Provider='{config.embeddings_provider}', "
        f"Model='{config.embeddings_model_name}', Dimensions='{config.embeddings_dimensions}', "
        f"InputType='{config.embeddings_document_input_type}', Encoding='{config.embeddings_encoding_format}'"
    )
    embedding_fn = EmbeddingModel(
        provider=config.embeddings_provider,
        model_name=config.embeddings_model_name,
        dimensions=config.embeddings_dimensions,
        input_type=config.embeddings_document_input_type,
        encoding_format=config.embeddings_encoding_format,
    )
    vectorstore_dir = config.vectordb_index_dir
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Vector store directory set to: {vectorstore_dir}")

    document_files: List[pathlib.Path] = list(
        pathlib.Path(artifact_dir).rglob("documents.jsonl")
    )
    logger.info(f"Found {len(document_files)} document file(s) in artifact.")

    transformed_documents = []
    for document_file in document_files:
        logger.debug(f"Loading documents from: {document_file}")
        with document_file.open() as f:
            for line in f:
                transformed_documents.append(Document(**json.loads(line)))
    logger.info(f"Loaded {len(transformed_documents)} total documents.")

    logger.info(
        f"Initializing Chroma vector store with collection: '{config.vectordb_collection_name}' "
        f"in directory: {config.vectordb_index_dir}"
    )
    chroma_client = ChromaVectorStore(
        embedding_model=embedding_fn,
        vector_store_config=config,
        chat_config=chat_config,
    )
    logger.info(f"Adding documents to Chroma in batches of {config.batch_size}...")
    for batch_idx in trange(0, len(transformed_documents), config.batch_size):
        batch = transformed_documents[batch_idx : batch_idx + config.batch_size]
        logger.debug(f"Adding batch {batch_idx // config.batch_size + 1} (size: {len(batch)}) to Chroma.")
        chroma_client.add_documents(batch)
    logger.info("Finished adding documents to Chroma.")

    logger.info(f"Creating result artifact: {result_artifact_name}")
    result_artifact = wandb.Artifact(
        name=result_artifact_name,
        type="vectorstore",
    )

    logger.info(f"Adding vector store directory '{config.vectordb_index_dir}' to artifact.")
    result_artifact.add_dir(
        local_path=str(config.vectordb_index_dir),
    )
    logger.info("Logging result artifact to W&B...")
    run.log_artifact(result_artifact, aliases=["chroma_index", "latest"])
    logger.info("Result artifact logged successfully.")

    run.finish()
    final_artifact_path = f"{entity}/{project}/{result_artifact_name}:latest"
    logger.info(f"Vector store creation finished. Final artifact: {final_artifact_path}")
    return final_artifact_path
