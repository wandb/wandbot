"""This module contains functions for loading and managing vector stores in the Wandbot ingestion system.

The module includes the following functions:
- `load`: Loads the vector store from the specified source artifact path and returns the name of the resulting artifact.

Typical usage example:

    project = "wandbot-dev"
    entity = "wandbot"
    source_artifact_path = "wandbot/wandbot-dev/raw_dataset:latest"
    load(project, entity, source_artifact_path)
"""

import json
import pathlib
from typing import List, Dict

from tqdm import trange

import wandb
from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.configs.chat_config import ChatConfig
from wandbot.models.embedding import EmbeddingModel
from wandbot.utils import get_logger
from wandbot.schema.document import Document
from wandbot.retriever.chroma import ChromaVectorStore
from wandbot.configs.ingestion_config import IngestionConfig

logger = get_logger(__name__)


# --- Helper Functions ---

def _download_source_artifact(run: wandb.sdk.wandb_run.Run, source_artifact_path: str) -> pathlib.Path:
    """Downloads the source artifact and returns the local directory path."""
    logger.info(f"Using source artifact: {source_artifact_path}")
    artifact: wandb.Artifact = run.use_artifact(source_artifact_path, type="dataset")
    logger.info("Downloading source artifact...")
    artifact_dir = artifact.download()
    logger.info(f"Source artifact downloaded to: {artifact_dir}")
    return pathlib.Path(artifact_dir)

def _load_and_count_documents(artifact_dir: pathlib.Path) -> (List[Document], Dict[str, int]):
    """Loads documents from jsonl files and counts them per source directory."""
    document_files: List[pathlib.Path] = list(artifact_dir.rglob("documents.jsonl"))
    logger.info(f"Found {len(document_files)} document file(s) in artifact.")
    source_doc_counts = {}
    transformed_documents = []
    logger.info("Loading documents and counting per source...")
    for document_file in document_files:
        source_name = document_file.parent.name
        count = 0
        logger.debug(f"Loading documents from: {document_file}")
        with document_file.open() as f:
            for line in f:
                transformed_documents.append(Document(**json.loads(line)))
                count += 1
        source_doc_counts[source_name] = count
        logger.debug(f"  Source '{source_name}' count: {count}")
    logger.info(f"Loaded {len(transformed_documents)} total documents.")
    logger.info(f"Source counts: {json.dumps(source_doc_counts, indent=2)}")
    return transformed_documents, source_doc_counts

def _add_documents_to_vectorstore(client: ChromaVectorStore, documents: List[Document], batch_size: int):
    """Adds documents to the vector store client in batches."""
    logger.info(f"Adding {len(documents)} documents to Chroma in batches of {batch_size}...")
    for batch_idx in trange(0, len(documents), batch_size):
        batch = documents[batch_idx : batch_idx + batch_size]
        logger.debug(
            f"Adding batch {batch_idx // batch_size + 1} (size: {len(batch)}) to Chroma."
        )
        client.add_documents(batch)
    logger.info("Finished adding documents to Chroma.")

# --- Main Function ---

def build_vector_store_artifact(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = None, # Allow overriding default name
) -> str:
    """Builds and logs the vector store artifact.

    Downloads the preprocessed data artifact, initializes embeddings and the vector store,
    adds documents, and logs the resulting vector store index as a new artifact.

    Args:
        project: The name of the project.
        entity: The name of the entity.
        source_artifact_path: The path to the source artifact (preprocessed data).
        result_artifact_name: Optional name for the resulting vector store artifact.

    Returns:
        The name of the resulting artifact.

    Raises:
        wandb.Error: An error occurred during the W&B operations.
        Exception: Other potential errors during processing.
    """
    logger.info(
        f"Starting vector store creation for {entity}/{project} from artifact {source_artifact_path}"
    )
    config: VectorStoreConfig = VectorStoreConfig()
    chat_config: ChatConfig = ChatConfig()
    ingestion_config: IngestionConfig = IngestionConfig()
    run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
        project=project, entity=entity, job_type="create_vectorstore"
    )
    if run is None:
        raise Exception("Failed to initialize wandb run.")

    # Use provided name or default from config
    final_artifact_name = result_artifact_name or ingestion_config.vectorstore_index_artifact_name
    
    try:
        # Download source artifact
        artifact_dir = _download_source_artifact(run, source_artifact_path)
        
        # Initialize embeddings
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
        
        # Ensure vectorstore directory exists
        vectorstore_dir = config.vectordb_index_dir
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Vector store directory set to: {vectorstore_dir}")

        # Load documents and count sources
        transformed_documents, source_doc_counts = _load_and_count_documents(artifact_dir)

        # Initialize vector store client
        logger.info(
            f"Initializing Chroma vector store with collection: '{config.vectordb_collection_name}' "
            f"in directory: {vectorstore_dir}"
        )
        chroma_client = ChromaVectorStore(
            embedding_model=embedding_fn,
            vector_store_config=config,
            chat_config=chat_config,
        )
        
        # Add documents
        _add_documents_to_vectorstore(chroma_client, transformed_documents, config.batch_size)
        
        # Create and log the result artifact
        logger.info(f"Creating result artifact: {final_artifact_name}")
        
        artifact_metadata = {
            "vector_store_config": config.model_dump(mode='json'), # Serialize config
            "source_document_counts": source_doc_counts,
            "total_documents_processed": len(transformed_documents)
        }
        logger.info(f"Artifact metadata prepared: {json.dumps(artifact_metadata, indent=2)}")

        result_artifact = wandb.Artifact(
            name=final_artifact_name,
            type=ingestion_config.vectorstore_index_artifact_type,
            metadata=artifact_metadata
        )

        logger.info(f"Adding vector store directory '{vectorstore_dir}' to artifact.")
        result_artifact.add_dir(local_path=str(vectorstore_dir))

        logger.info("Logging result artifact to W&B...")
        aliases = [
            "latest",
            f"embed-model-{config.embeddings_model_name}",
            f"embed-dim-{config.embeddings_dimensions}",
        ]
        run.log_artifact(result_artifact, aliases=aliases)
        logger.info("Result artifact logged successfully.")

        final_artifact_path = f"{entity}/{project}/{final_artifact_name}:latest"
        logger.info(f"Vector store creation finished. Final artifact: {final_artifact_path}")
        return final_artifact_path

    except Exception as e:
        logger.error(f"Vectorstore pipeline failed: {e}", exc_info=True)
        if run:
            run.finish(exit_code=1)
        raise
    finally:
        if run and run.state == "running":
            run.finish()
            logger.info(f"Wandb run finished: {run.url}")
