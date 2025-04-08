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
    """Loads documents (text chunks) from jsonl files and counts them per source directory."""
    document_files: List[pathlib.Path] = list(artifact_dir.rglob("documents.jsonl"))
    logger.info(f"Found {len(document_files)} preprocessed document file(s) in artifact.")
    source_doc_counts = {}
    transformed_documents = []
    logger.info("Loading preprocessed documents (text chunks) and counting per source...")
    for document_file in document_files:
        source_name = document_file.parent.name
        count = 0
        logger.debug(f"Loading text chunks from: {document_file}")
        with document_file.open() as f:
            for line in f:
                transformed_documents.append(Document(**json.loads(line)))
                count += 1
        source_doc_counts[source_name] = count
        logger.debug(f"  Source '{source_name}' count (text chunks): {count}")
    logger.info(f"Loaded {len(transformed_documents)} total text chunks.")
    logger.info(f"Source counts (text chunks): {json.dumps(source_doc_counts, indent=2)}")
    return transformed_documents, source_doc_counts

def _add_documents_to_vectorstore(client: ChromaVectorStore, documents: List[Document], batch_size: int):
    """Adds preprocessed documents (text chunks) to the vector store client in batches."""
    logger.info(f"Adding {len(documents)} preprocessed documents (text chunks) to Chroma in batches of {batch_size}...")
    for batch_idx in trange(0, len(documents), batch_size):
        batch = documents[batch_idx : batch_idx + batch_size]
        logger.debug(
            f"Adding batch {batch_idx // batch_size + 1} (size: {len(batch)} text chunks) to Chroma."
        )
        client.add_documents(batch)
    logger.info("Finished adding text chunks to Chroma.")

# --- Main Function ---

def build_vector_store_artifact(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = None, # Allow overriding default name
    debug: bool = False, # Add debug flag
) -> str:
    """Builds and logs the vector store artifact.

    Downloads the preprocessed data artifact, initializes embeddings and the vector store,
    adds documents, and logs the resulting vector store index as a new artifact.

    Args:
        project: The name of the project.
        entity: The name of the entity.
        source_artifact_path: The path to the source artifact (preprocessed data).
        result_artifact_name: Optional name for the resulting vector store artifact.
        debug: If True, indicates a debug run.

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
        
        # Serialize config once
        serialized_config = config.model_dump(mode='json')

        # Keys to remove from the serialized config in metadata
        keys_to_remove = [
            "vectordb_index_artifact_url", # Often derived or set elsewhere
            "vector_store_auth_token",    # Sensitive
            "embeddings_query_input_type", # Redundant if embeddings_document_input_type exists
            # Keep embeddings_document_input_type as it's relevant context
        ]
        for key in keys_to_remove:
            serialized_config.pop(key, None)

        artifact_metadata = {
            "vector_store_config": serialized_config, # Use modified config
            "source_document_counts": source_doc_counts,
            "total_documents_processed": len(transformed_documents)
        }
        logger.info(f"Artifact metadata prepared: {json.dumps(artifact_metadata, indent=2)}")

        # Prepare description
        description_string = f"Chroma vector store artifact for {entity}/{project}.\\n"
        description_string += f"Built from source artifact: {source_artifact_path}\\n"
        description_string += f"Contains {len(transformed_documents)} embedded text chunks."
        if debug:
            description_string += " (DEBUG MODE: Data potentially limited)"
        description_string += "\\n\\nMetadata details:\\n"
        description_string += json.dumps(artifact_metadata, indent=2)

        result_artifact = wandb.Artifact(
            name=final_artifact_name,
            type=ingestion_config.vectorstore_index_artifact_type,
            metadata=artifact_metadata,
            description=description_string # Add description here
        )

        logger.info(f"Adding vector store directory '{vectorstore_dir}' to artifact.")
        result_artifact.add_dir(local_path=str(vectorstore_dir))

        logger.info("Logging result artifact to W&B...")
        aliases = [
            f"embed-model_{config.embeddings_model_name}",
            f"embed-dim_{config.embeddings_dimensions}",
        ]
        if debug:
            aliases.append("debug") # Add a debug alias

        run.log_artifact(result_artifact, aliases=aliases) # Metadata & description already set
        logger.info(f"Artifact {result_artifact.name} logged with aliases: {aliases}, now uploading...")
        result_artifact.wait() # Wait for the upload to complete
        logger.info(f"Artifact {result_artifact.name} upload complete.")

        # Now, fetch the logged artifact and add tags, with error handling
        try:
            api = wandb.Api()
            # Use the artifact name and 'latest' alias to fetch the version just logged
            logged_artifact_name = f"{entity}/{project}/{final_artifact_name}:latest"
            logged_artifact = api.artifact(logged_artifact_name, type=ingestion_config.vectorstore_index_artifact_type)

            # Add tags (aliases + potentially 'debug') to the logged artifact
            # Ensure uniqueness
            existing_tags = set(logged_artifact.tags or [])
            new_tags = set(aliases) # Aliases already include 'debug' if needed
            logged_artifact.tags = list(existing_tags.union(new_tags))
            logged_artifact.save()
            logger.info(f"Successfully added tags {list(new_tags)} to artifact {logged_artifact_name}")
        except Exception as e:
            # Log a warning instead of erroring out if tag addition fails
            logger.warning(f"Failed to add tags {aliases} to artifact {final_artifact_name}: {e}", exc_info=False)

        final_artifact_path = f"{entity}/{project}/{final_artifact_name}:latest"
        logger.info(f"Vector store creation finished. Final artifact: {final_artifact_path}")
        return final_artifact_path

    except Exception as e:
        logger.error(f"Vectorstore pipeline failed: {e}", exc_info=True)
        if run:
            run.finish(exit_code=1)
        raise
    finally:
        # Ensure the run is finished cleanly
        if run:
            run.finish()
            logger.info(f"Wandb run finished: {run.url}")
