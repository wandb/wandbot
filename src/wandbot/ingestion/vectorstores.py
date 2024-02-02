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
from typing import Any, Dict, List

import wandb
from langchain.schema import Document as LcDocument
from llama_index.schema import TextNode
from wandbot.ingestion import preprocess_data
from wandbot.ingestion.config import VectorStoreConfig
from wandbot.utils import (
    get_logger,
    load_index,
    load_service_context,
    load_storage_context,
)

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
    service_context = load_service_context(
        embeddings_model=config.embeddings_model,
        embeddings_size=config.embedding_dim,
        llm="gpt-3.5-turbo-16k-0613",
        temperature=config.temperature,
        max_retries=config.max_retries,
    )

    storage_context = load_storage_context(persist_dir=str(config.persist_dir))

    document_files: List[pathlib.Path] = list(
        pathlib.Path(artifact_dir).rglob("documents.jsonl")
    )

    transformed_documents: List[TextNode] = []
    indices = []
    for document_file in document_files:
        documents: List[LcDocument] = []
        with document_file.open() as f:
            for line in f:
                doc_dict: Dict[str, Any] = json.loads(line)
                doc: LcDocument = LcDocument(**doc_dict)
                documents.append(doc)
        preprocessed_documents = preprocess_data.load(documents)
        unique_objects = {obj.hash: obj for obj in preprocessed_documents}
        preprocessed_documents = list(unique_objects.values())

        for document in preprocessed_documents:
            document.metadata["index"] = document_file.parent.name
            tags_list = (
                document.metadata["tags"] if document.metadata["tags"] else []
            )

            if tags_list:
                document.metadata["tags"] = ",".join(tags_list)
            else:
                document.metadata["tags"] = ""

        transformed_documents.extend(preprocessed_documents)
        indices.append(document_file.parent.name)

    logger.info(f"Number of documents: {len(transformed_documents)}")
    _ = load_index(
        transformed_documents,
        service_context,
        storage_context,
        persist_dir=str(config.persist_dir),
    )
    artifact = wandb.Artifact(
        name="wandbot_index",
        type="storage_context",
        metadata={"indices": indices},
    )
    artifact.add_dir(
        local_path=str(config.persist_dir),
    )
    run.log_artifact(artifact)

    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"
