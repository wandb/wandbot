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

from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import trange

import wandb
from wandbot.ingestion.config import VectorStoreConfig
from wandbot.utils import get_logger

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
    config: VectorStoreConfig = VectorStoreConfig()
    run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
        project=project, entity=entity, job_type="create_vectorstore"
    )
    artifact: wandb.Artifact = run.use_artifact(
        source_artifact_path, type="dataset"
    )
    artifact_dir: str = artifact.download()

    embedding_fn = OpenAIEmbeddings(
        model=config.embeddings_model, dimensions=config.embedding_dim
    )
    vectorstore_dir = config.persist_dir
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    document_files: List[pathlib.Path] = list(
        pathlib.Path(artifact_dir).rglob("documents.jsonl")
    )

    transformed_documents = []
    for document_file in document_files:
        with document_file.open() as f:
            for line in f:
                transformed_documents.append(Document(**json.loads(line)))

    chroma = Chroma(
        collection_name=config.name,
        embedding_function=embedding_fn,
        persist_directory=str(config.persist_dir),
    )
    for batch_idx in trange(0, len(transformed_documents), config.batch_size):
        batch = transformed_documents[batch_idx : batch_idx + config.batch_size]
        chroma.add_documents(batch)
    chroma.persist()

    result_artifact = wandb.Artifact(name="chroma_index", type="vectorstore")

    result_artifact.add_dir(
        local_path=str(config.persist_dir),
    )
    run.log_artifact(result_artifact)

    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"
