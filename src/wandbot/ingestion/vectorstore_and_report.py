import copy
import json
import math
import os
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from tqdm import trange
from chromadb.config import Settings

import wandb
import wandb.apis.reports as wr
from wandbot.configs.chat_config import ChatConfig
from wandbot.configs.ingestion_config import IngestionConfig
from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.models.embedding import EmbeddingModel
from wandbot.retriever.chroma import ChromaVectorStore
from wandbot.schema.document import Document
from wandbot.utils import get_logger

logger = get_logger(__name__)

# --- Helper Functions (Moved from other modules) ---

def _get_artifact_base_name(artifact_full_name: str) -> str:
    """Extracts the base name (without version/alias) from an artifact path."""
    return artifact_full_name.split("/")[-1].split(":")[0]

def _log_raw_counts(metadata: dict[str, dict[str, int]]) -> list[str]:
    """Logs the number of documents for each data source.
       (Moved from report.py)
    """
    data: dict[str, int] = {}
    for source, info in metadata.items():
        data[source] = info["num_documents"]
    if wandb.run:
        wandb.run.log(data)
    else:
        logger.warning("No active W&B run to log raw counts.")
    return list(data.keys())

def _get_raw_metadata_from_artifact(run: wandb.sdk.wandb_run.Run, raw_artifact_path: str) -> dict[str, dict[str, int]]:
    """Extracts metadata from the raw data artifact.
       (Moved from report.py)
    """
    raw_artifact = run.use_artifact(raw_artifact_path, type="dataset")
    raw_artifact_dir = raw_artifact.download()
    raw_metadata_files = list(pathlib.Path(raw_artifact_dir).rglob("metadata.json"))
    raw_metadata: dict[str, dict[str, int]] = {}
    for metadata_file in raw_metadata_files:
        with metadata_file.open("r") as f:
            raw_metadata[metadata_file.parent.name] = json.load(f)
    return raw_metadata

def _download_source_artifact(run: wandb.sdk.wandb_run.Run, source_artifact_path: str) -> pathlib.Path:
    """Downloads the source artifact and returns the local directory path.
       (Moved from create_vector_store.py)
    """
    logger.info(f"Using source artifact: {source_artifact_path}")
    artifact: wandb.Artifact = run.use_artifact(source_artifact_path, type="dataset")
    logger.info("Downloading source artifact...")
    artifact_dir = artifact.download()
    logger.info(f"Source artifact downloaded to: {artifact_dir}")
    return pathlib.Path(artifact_dir)

def _load_and_count_documents(artifact_dir: pathlib.Path) -> Tuple[List[Document], Dict[str, int]]:
    """Loads documents (text chunks) from jsonl files and counts them per source directory.
       (Moved from create_vector_store.py)
    """
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
    """Adds preprocessed documents (text chunks) to the vector store client in batches.
       (Moved from create_vector_store.py)
    """
    logger.info(f"Adding {len(documents)} preprocessed documents (text chunks) to Chroma in batches of {batch_size}...")
    for batch_idx in trange(0, len(documents), batch_size):
        batch = documents[batch_idx : batch_idx + batch_size]
        logger.debug(
            f"Adding batch {batch_idx // batch_size + 1} (size: {len(batch)} text chunks) to Chroma."
        )
        logger.debug(f"First document in batch: {batch[0]}")
        client.add_documents(batch)
    logger.info("Finished adding text chunks to Chroma.")

def _log_chunk_counts(metadata: dict[str, int]) -> list[str]:
    """Logs the number of chunks for each data source to the current run."""
    # Prefix metrics to avoid collision with raw counts
    chunk_metrics = {f"chunks_{source}": count for source, count in metadata.items()}
    wandb.log(chunk_metrics) 
    return list(chunk_metrics.keys())

# --- Remote ChromaDB Upload Helpers ---

def _connect_remote_chroma(vs_config: VectorStoreConfig) -> Optional[chromadb.HttpClient]:
    """Connects to the remote ChromaDB instance using existing config."""
    # Use existing config fields
    host = vs_config.vector_store_host
    tenant = vs_config.vector_store_tenant
    database = vs_config.vector_store_database
    api_key = vs_config.vector_store_api_key # Assumes this is loaded from env correctly

    if not all([host, tenant, database]):
        logger.warning(
            "Missing remote ChromaDB configuration (vector_store_host, vector_store_tenant, or vector_store_database). Skipping upload."
        )
        return None

    logger.info(
        f"Connecting to remote ChromaDB: host={host}, " 
        f"tenant={tenant}, database={database}"
    )
    
    headers = {}
    if api_key:
        logger.info("Using API key for remote ChromaDB connection.")
        headers['x-chroma-token'] = api_key
    else:
        logger.warning("No vector_store_api_key found in config. Connecting without authentication header.")
        
    try:
        # Mimic HttpClient instantiation from chroma.py
        remote_client = chromadb.HttpClient(
            host=host,
            ssl=True, # Assuming SSL is always true for hosted
            tenant=tenant,
            database=database,
            headers=headers,
            settings=Settings(anonymized_telemetry=False) # Match settings from chroma.py
        )
        remote_client.heartbeat() # Test connection
        logger.info("Remote ChromaDB client connected.")
        return remote_client
    except Exception as e:
        logger.error(f"Error connecting to remote ChromaDB client: {e}", exc_info=True)
        return None

def _transform_and_clean_data_for_upload(
    local_data: Dict[str, List[Any]],
    vs_config: VectorStoreConfig,
) -> Tuple[List[str], List[Optional[Dict]]]:
    """Prepares documents and metadata for remote upload."""
    processed_documents = []
    processed_metadatas = []

    for idx in range(len(local_data['ids'])):
        doc = local_data['documents'][idx]
        meta = local_data['metadatas'][idx]

        new_doc_parts = []
        metadata_prepended_content = []
        cleaned_meta = copy.deepcopy(meta) if meta else {}

        if meta:
            for key_to_prepend in vs_config.remote_chroma_keys_to_prepend:
                value = meta.get(key_to_prepend)
                if value:
                    metadata_prepended_content.append(f"{key_to_prepend.capitalize()}: {value}\n")

            for key_to_remove in vs_config.remote_chroma_keys_to_remove:
                cleaned_meta.pop(key_to_remove, None)

        if metadata_prepended_content:
            new_doc_parts.append("--- Metadata ---\n")
            new_doc_parts.extend(metadata_prepended_content)
            new_doc_parts.append("\n--- Document ---")

        new_doc_parts.append(doc)

        processed_documents.append("\n".join(new_doc_parts))
        processed_metadatas.append(cleaned_meta if cleaned_meta else None)

    return processed_documents, processed_metadatas

def _upload_to_remote_chroma(
    local_client: ChromaVectorStore,
    remote_client: chromadb.HttpClient,
    remote_collection_name: str,
    vs_config: VectorStoreConfig,
):
    """Fetches data from local Chroma and uploads to remote ChromaDB."""
    logger.info(f"Starting upload to remote collection: {remote_collection_name}")

    try:
        # The ChromaVectorStore wrapper manages a single collection internally.
        # Access it directly via the 'collection' attribute.
        local_collection = local_client.collection
        if not local_collection:
            logger.warning("Local client does not have an initialized collection.")
            return
        local_collection_name = local_collection.name

        logger.info(f"Fetching all data from local collection '{local_collection_name}'...")
        # Note: Fetching all data might be memory-intensive for very large collections.
        # Consider streaming/batching fetches if needed.
        local_data = local_collection.get(include=['embeddings', 'documents', 'metadatas'])
        item_count = len(local_data['ids'])
        logger.info(f"Fetched {item_count} items from local collection.")

        if item_count == 0:
            logger.info(f"Local collection '{local_collection_name}' is empty. Skipping upload.")
            return

        logger.info(f"Getting or creating remote collection '{remote_collection_name}'...")
        remote_collection = remote_client.get_or_create_collection(
            name=remote_collection_name,
            metadata={vs_config.distance_key: vs_config.distance} # Use distance metric from config
        )
        logger.info(f"Got remote collection '{remote_collection_name}'.")

        logger.info("Transforming documents and cleaning metadata for upload...")
        processed_docs, processed_metas = _transform_and_clean_data_for_upload(
            local_data, vs_config
        )

        num_batches = math.ceil(item_count / vs_config.batch_size)
        logger.info(
            f"Upserting {item_count} items to remote collection '{remote_collection_name}' in {num_batches} batches of size {vs_config.batch_size}..."
        )

        total_items_processed = 0
        for i in trange(0, item_count, vs_config.batch_size):
            batch_num = (i // vs_config.batch_size) + 1
            start_index = i
            end_index = min(i + vs_config.batch_size, item_count)

            ids_batch = local_data['ids'][start_index:end_index]
            embeddings_batch = local_data['embeddings'][start_index:end_index]
            docs_batch = processed_docs[start_index:end_index]
            metas_batch = processed_metas[start_index:end_index]

            logger.debug(f"  -> Upserting batch {batch_num}/{num_batches} (items {start_index+1}-{end_index})...")
            try:
                remote_collection.upsert(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=docs_batch,
                    metadatas=metas_batch,
                )
                logger.debug(f"  -> Batch {batch_num} upserted.")
                total_items_processed += len(ids_batch)
            except Exception as batch_e:
                logger.error(
                    f"Error upserting batch {batch_num} (items {start_index+1}-{end_index}) to remote collection '{remote_collection_name}': {batch_e}",
                    exc_info=True
                )
                trace_id = "N/A"
                if "trace ID:" in str(batch_e):
                    try:
                        trace_id = str(batch_e).split("trace ID:")[1].split(")")[0].strip()
                    except IndexError:
                        pass
                logger.error(f"Trace ID (if available in error): {trace_id}")
                logger.warning(f"Skipping remainder of upload for collection '{remote_collection_name}' due to batch error.")
                break # Stop upload for this collection on batch error

        logger.info(f"Finished upload attempt for collection '{remote_collection_name}'. Total items processed in successful batches: {total_items_processed}")

    except Exception as e:
        logger.error(
            f"Failed to upload data to remote ChromaDB collection '{remote_collection_name}': {e}",
            exc_info=True
        )

# --- Main Pipeline Function ---

def run_vectorstore_and_report_pipeline(
    project: str,
    entity: str,
    raw_artifact_path: str,
    preprocessed_artifact_path: str,
    vectorstore_artifact_name: str,
    debug: bool = False,
    create_report: bool = True,
    upload_to_remote_vector_store: bool = True,
) -> str:
    """
    Builds the vector store, logs it as an artifact, optionally uploads to remote Chroma,
    and optionally creates a W&B Report summarizing the ingestion process,
    all within a single W&B run.

    Args:
        project: The W&B project name.
        entity: The W&B entity name.
        raw_artifact_path: Full path to the raw data artifact.
        preprocessed_artifact_path: Full path to the preprocessed data artifact.
        vectorstore_artifact_name: Desired base name for the vector store artifact.
        debug: If True, indicates a debug run.
        create_report: If True, generates and saves a W&B report.
        upload_to_remote_vector_store: If True, attempts to upload the collection to remote ChromaDB.

    Returns:
        The full path of the logged vector store artifact.

    Raises:
        Exception: If any step in the process fails.
    """
    run = None
    remote_chroma_client = None
    final_vs_artifact_path = None
    local_chroma_client = None
    logged_artifact_version = None
    try:
        logger.info(
            f"Starting combined Vector Store and Report pipeline for {entity}/{project}"
        )
        vs_config: VectorStoreConfig = VectorStoreConfig()
        chat_config: ChatConfig = ChatConfig()
        ingestion_config: IngestionConfig = IngestionConfig()

        # Initialize a single W&B run for both steps
        run = wandb.init(project=project, entity=entity, job_type="vectorstore_report")
        if run is None:
            raise Exception("Failed to initialize wandb run.")
        logger.info(f"W&B Run initialized: {run.url}")

        # --- Vector Store Creation Logic (Adapted) ---
        logger.info(f"Building vector store from: {preprocessed_artifact_path}")
        artifact_dir = _download_source_artifact(run, preprocessed_artifact_path)

        embedding_fn = EmbeddingModel(
            provider=vs_config.embeddings_provider,
            model_name=vs_config.embeddings_model_name,
            dimensions=vs_config.embeddings_dimensions,
            input_type=vs_config.embeddings_document_input_type,
            encoding_format=vs_config.embeddings_encoding_format,
        )

        # Ensure local persist directory exists and is *unique* for this run
        # This prevents interference if multiple runs use the same base directory.
        # We'll use the run ID to make it unique.
        base_persist_dir = vs_config.vectordb_index_dir
        local_persist_dir = base_persist_dir / f"run_{run.id}"
        local_persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using local temporary persist directory: {local_persist_dir}")

        # Adjust VectorStoreConfig to use the temporary local path for persistence
        temp_vs_config = vs_config.model_copy(update={"persist_directory": local_persist_dir})

        transformed_documents, source_doc_counts = _load_and_count_documents(artifact_dir)

        local_chroma_client = ChromaVectorStore(
            embedding_model=embedding_fn,
            vector_store_config=temp_vs_config, # Use config with temp path
            chat_config=chat_config,
        )

        _add_documents_to_vectorstore(
            local_chroma_client, transformed_documents, vs_config.batch_size
        )

        # Persist the local ChromaDB data explicitly
        logger.info(f"Persisting local ChromaDB client to {local_persist_dir}...")
        # Access the underlying chromadb client and call persist. The method might vary.
        # Assuming the client object is accessible via local_chroma_client.client
        if hasattr(local_chroma_client, 'client') and hasattr(local_chroma_client.client, 'persist'):
             local_chroma_client.client.persist()
             logger.info("Local ChromaDB persisted.")
        else:
             logger.warning("Could not find a 'persist' method on the local Chroma client. Data might not be saved correctly for artifacting.")

        logger.info(f"Creating vector store artifact: {vectorstore_artifact_name}")
        # Create metadata *before* removing keys for artifact logging
        vs_artifact_metadata = {
            "vector_store_config": vs_config.model_dump(mode='json'), # Log original config
            "source_document_counts": source_doc_counts,  # Chunks per source
            "total_documents_processed": len(transformed_documents),  # Total chunks
        }

        description_string = f"Chroma vector store artifact for {entity}/{project}.\n"
        description_string += (
            f"Built from preprocessed artifact: {preprocessed_artifact_path}\n"
        )
        description_string += (
            f"Contains {len(transformed_documents)} embedded text chunks."
        )
        if debug:
            description_string += " (DEBUG MODE: Data potentially limited)"
        description_string += "\n\nMetadata details:\n"
        description_string += json.dumps(vs_artifact_metadata, indent=2)

        result_artifact = wandb.Artifact(
            name=vectorstore_artifact_name,
            type=ingestion_config.vectorstore_index_artifact_type,
            metadata=vs_artifact_metadata,
            description=description_string,
        )

        # Add the *persisted directory* content to the artifact
        result_artifact.add_dir(local_path=str(local_persist_dir))

        logger.info("Logging vector store artifact to W&B...")
        aliases = [
            f"embed-model_{vs_config.embeddings_model_name}",
            f"embed-dim_{vs_config.embeddings_dimensions}",
        ]
        if debug:
            aliases.append("debug")

        log_result = run.log_artifact(result_artifact, aliases=aliases)
        log_result.wait()  # Wait for the artifact upload to complete
        logged_artifact_version = log_result.version
        logger.info(
            f"Artifact {result_artifact.name} logged with aliases: {aliases}, version: {logged_artifact_version}, uploading..."
        )
        final_vs_artifact_path = (
            f"{entity}/{project}/{vectorstore_artifact_name}:v{logged_artifact_version}"
        )
        logger.info(f"Vector store artifact upload complete: {final_vs_artifact_path}")

        # --- Remote ChromaDB Upload (Conditional) ---
        if upload_to_remote_vector_store:
            logger.info("Attempting upload to remote ChromaDB...")
            remote_chroma_client = _connect_remote_chroma(vs_config)
            if remote_chroma_client and local_chroma_client and logged_artifact_version:
                # Construct remote collection name: artifact_name-v<version>(-debug)
                remote_collection_name = f"{vectorstore_artifact_name}-v{logged_artifact_version}"
                if debug:
                    remote_collection_name += "-debug"

                _upload_to_remote_chroma(
                    local_chroma_client, # Pass the local client instance
                    remote_chroma_client,
                    remote_collection_name,
                    vs_config,
                )
            else:
                logger.warning("Skipping remote ChromaDB upload due to connection failure or missing local client/artifact version.")
        else:
            logger.info("Skipping upload to remote ChromaDB as per configuration.")

        # --- Report Creation Logic (Conditional) ---
        if create_report:
            logger.info("Starting report creation within the same run...")
            report_title = f"Wandbot Data Ingestion Report ({vectorstore_artifact_name}-v{logged_artifact_version}): {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            report_description = f"Vector store creation report for {run.name}."
            if debug:
                report_title += " (DEBUG RUN)"
                report_description += " (DEBUG MODE)"

            report = wr.Report(
                project=run.project,
                entity=run.entity,
                title=report_title,
                description=report_description,
            )

            # Fetch raw metadata using the same run
            logger.info(f"Fetching metadata from raw artifact: {raw_artifact_path}")
            raw_metadata = _get_raw_metadata_from_artifact(
                run, raw_artifact_path
            )  # Use local helper
            raw_sources = _log_raw_counts(raw_metadata)  # Use local helper

            # Log chunk counts and create the plot for chunks
            chunk_counts = vs_artifact_metadata.get("source_document_counts", {})
            pg_chunks = None  # Initialize pg_chunks
            if chunk_counts:
                chunk_sources_metrics = _log_chunk_counts(chunk_counts)
                pg_chunks = wr.PanelGrid(
                    runsets=[
                        wr.Runset(run.entity, run.project, query=run.name),
                    ],
                    panels=[
                        wr.BarPlot(
                            title="Data Sources (Chunks)", metrics=chunk_sources_metrics
                        )
                    ],
                )
            else:
                logger.warning(
                    "Chunk counts per source not found in metadata, skipping chunk plot."
                )

            pg_raw = wr.PanelGrid(
                runsets=[wr.Runset(run.entity, run.project, query=run.name)],
                panels=[wr.BarPlot(title="Data Sources (Raw Docs)", metrics=raw_sources)],
            )

            report.blocks = [
                wr.TableOfContents(),
                wr.H1("Run Information"),
                wr.P(f"Run Details: [View Run]({run.url})"),  # Link to the current run
                wr.H1("Vector Store"),
                wr.H2("Vector Store Chunk Counts"),
                wr.P("Chunk counts per source:"),
            ]
            # Conditionally add chunk plot if it was created
            if pg_chunks:
                report.blocks.append(pg_chunks)

            report.blocks.extend(
                [
                    wr.H2("Vector Store Artifact Metadata"),
                    wr.CodeBlock(
                        [json.dumps(vs_artifact_metadata, indent=2)], language="json"
                    ),
                    wr.P(
                        f"Vector store built from artifact: `{_get_artifact_base_name(preprocessed_artifact_path)}`"
                    ),
                    wr.P(f"Logged artifact: `{final_vs_artifact_path}`"), # Use final path with version
                    wr.WeaveBlockArtifact(
                        run.entity,
                        run.project,
                        f"{vectorstore_artifact_name}:v{logged_artifact_version}", # Use name + version
                        "overview",
                    ),
                    wr.H1("Remote ChromaDB Upload"),
                    wr.P("Upload to remote ChromaDB was attempted." if upload_to_remote_vector_store else "Upload to remote ChromaDB was skipped."),
                    wr.P(f"Remote Collection Name (if uploaded): `{remote_collection_name}`") if upload_to_remote_vector_store and logged_artifact_version else wr.P(""),
                    wr.H1("Raw Data Sources"),
                    wr.P(
                        f"Raw data loaded from artifact: `{_get_artifact_base_name(raw_artifact_path)}`"
                    ),
                    wr.H1("Raw Datasources Metadata (Document Counts)"),
                    wr.UnorderedList(list(raw_metadata.keys())),
                    pg_raw,
                    wr.CodeBlock([json.dumps(raw_metadata, indent=2)], language="json"),
                ]
            )

            report.save()
            logger.info(f"Report saved: {report.url}")
        else:
            logger.info("Skipping report creation as per configuration.")

        # Clean up temporary local directory after artifact logging and potential upload
        # try:
        #     if local_persist_dir.exists():
        #         logger.info(f"Cleaning up temporary directory: {local_persist_dir}")
        #         shutil.rmtree(local_persist_dir)
        # except Exception as cleanup_e:
        #     logger.warning(f"Failed to clean up temporary directory {local_persist_dir}: {cleanup_e}")

        return final_vs_artifact_path

    except Exception as e:
        logger.error(
            f"Combined vectorstore and report pipeline failed: {e}", exc_info=True
        )
        if run:
            run.finish(exit_code=1)
        raise
    finally:
        if run:
            run.finish()
            logger.info(f"W&B Run finished: {run.url}") 