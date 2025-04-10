import json
import pathlib
from typing import Dict, List, Any, Tuple

import wandb
import wandb.apis.reports as wr
from tqdm import trange
from datetime import datetime

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
        client.add_documents(batch)
    logger.info("Finished adding text chunks to Chroma.")

def _log_chunk_counts(metadata: dict[str, int]) -> list[str]:
    """Logs the number of chunks for each data source to the current run."""
    # Prefix metrics to avoid collision with raw counts
    chunk_metrics = {f"chunks_{source}": count for source, count in metadata.items()}
    wandb.log(chunk_metrics) 
    return list(chunk_metrics.keys())

# --- Main Pipeline Function ---

def run_vectorstore_and_report_pipeline(
    project: str,
    entity: str,
    raw_artifact_path: str,
    preprocessed_artifact_path: str,
    vectorstore_artifact_name: str,
    debug: bool = False,
    create_report: bool = True,
) -> str:
    """
    Builds the vector store, logs it as an artifact, and optionally creates a W&B Report
    summarizing the ingestion process, all within a single W&B run.

    Args:
        project: The W&B project name.
        entity: The W&B entity name.
        raw_artifact_path: Full path to the raw data artifact.
        preprocessed_artifact_path: Full path to the preprocessed data artifact.
        vectorstore_artifact_name: Desired base name for the vector store artifact.
        debug: If True, indicates a debug run.
        create_report: If True, generates and saves a W&B report.

    Returns:
        The full path of the logged vector store artifact.

    Raises:
        Exception: If any step in the process fails.
    """
    run = None
    try:
        logger.info(f"Starting combined Vector Store and Report pipeline for {entity}/{project}")
        vs_config: VectorStoreConfig = VectorStoreConfig()
        chat_config: ChatConfig = ChatConfig()
        ingestion_config: IngestionConfig = IngestionConfig()

        # Initialize a single W&B run for both steps
        run = wandb.init(
            project=project, entity=entity, job_type="vectorstore_report"
        )
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

        vectorstore_dir = vs_config.vectordb_index_dir
        vectorstore_dir.mkdir(parents=True, exist_ok=True)

        transformed_documents, source_doc_counts = _load_and_count_documents(artifact_dir)

        chroma_client = ChromaVectorStore(
            embedding_model=embedding_fn,
            vector_store_config=vs_config,
            chat_config=chat_config,
        )

        _add_documents_to_vectorstore(chroma_client, transformed_documents, vs_config.batch_size)

        logger.info(f"Creating vector store artifact: {vectorstore_artifact_name}")
        serialized_config = vs_config.model_dump(mode='json')
        keys_to_remove = [
            "vectordb_index_artifact_url", "vector_store_auth_token", "embeddings_query_input_type",
        ]
        for key in keys_to_remove:
            serialized_config.pop(key, None)

        vs_artifact_metadata = {
            "vector_store_config": serialized_config,
            "source_document_counts": source_doc_counts, # Chunks per source
            "total_documents_processed": len(transformed_documents) # Total chunks
        }

        description_string = f"Chroma vector store artifact for {entity}/{project}.\\n"
        description_string += f"Built from preprocessed artifact: {preprocessed_artifact_path}\\n"
        description_string += f"Contains {len(transformed_documents)} embedded text chunks."
        if debug:
            description_string += " (DEBUG MODE: Data potentially limited)"
        description_string += "\\n\\nMetadata details:\\n"
        description_string += json.dumps(vs_artifact_metadata, indent=2)

        result_artifact = wandb.Artifact(
            name=vectorstore_artifact_name,
            type=ingestion_config.vectorstore_index_artifact_type,
            metadata=vs_artifact_metadata,
            description=description_string
        )

        result_artifact.add_dir(local_path=str(vectorstore_dir))

        logger.info("Logging vector store artifact to W&B...")
        aliases = [
            f"embed-model_{vs_config.embeddings_model_name}",
            f"embed-dim_{vs_config.embeddings_dimensions}",
        ]
        if debug:
            aliases.append("debug")

        run.log_artifact(result_artifact, aliases=aliases)
        logger.info(f"Artifact {result_artifact.name} logged with aliases: {aliases}, uploading...")
        result_artifact.wait()
        final_vs_artifact_path = f"{entity}/{project}/{vectorstore_artifact_name}:latest"
        logger.info(f"Vector store artifact upload complete: {final_vs_artifact_path}")

        # --- Report Creation Logic (Conditional) ---
        if create_report:
            logger.info("Starting report creation within the same run...")
            report_title = f"Wandbot Data Ingestion Report for ({result_artifact.name}): {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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
            raw_metadata = _get_raw_metadata_from_artifact(run, raw_artifact_path) # Use local helper
            raw_sources = _log_raw_counts(raw_metadata) # Use local helper

            # Log chunk counts and create the plot for chunks
            chunk_counts = vs_artifact_metadata.get("source_document_counts", {})
            pg_chunks = None # Initialize pg_chunks
            if chunk_counts:
                chunk_sources_metrics = _log_chunk_counts(chunk_counts)
                pg_chunks = wr.PanelGrid(
                    runsets=[
                        wr.Runset(run.entity, run.project, query=run.name),
                    ],
                    panels=[
                        wr.BarPlot(title="Data Sources (Chunks)", metrics=chunk_sources_metrics)
                    ],
                )
            else:
                logger.warning("Chunk counts per source not found in metadata, skipping chunk plot.")

            pg_raw = wr.PanelGrid(
                runsets=[wr.Runset(run.entity, run.project, query=run.name)],
                panels=[wr.BarPlot(title="Data Sources (Raw Docs)", metrics=raw_sources)],
            )

            report.blocks = [
                wr.TableOfContents(),
                wr.H1("Run Information"),
                wr.P(f"Run Details: [View Run]({run.url})"), # Link to the current run
                wr.H1("Vector Store"),
                wr.H2("Vector Store Chunk Counts"),
                wr.P("Chunk counts per source:"),
            ]
            # Conditionally add chunk plot if it was created
            if pg_chunks:
                report.blocks.append(pg_chunks)
                
            report.blocks.extend([
                wr.H2("Vector Store Artifact Metadata"),
                wr.CodeBlock([json.dumps(vs_artifact_metadata, indent=2)], language="json"),
                wr.P(f"Vector store built from artifact: `{_get_artifact_base_name(preprocessed_artifact_path)}`"),
                wr.P(f"Logged artifact: `{vectorstore_artifact_name}`"),
                wr.WeaveBlockArtifact(
                    run.entity, run.project, vectorstore_artifact_name, "overview"
                ),
                wr.H1("Raw Data Sources"),
                wr.P(f"Raw data loaded from artifact: `{_get_artifact_base_name(raw_artifact_path)}`"),
                wr.H1("Raw Datasources Metadata (Document Counts)"),
                wr.UnorderedList(list(raw_metadata.keys())),
                pg_raw,
                wr.CodeBlock([json.dumps(raw_metadata, indent=2)], language="json"),
            ])

            report.save()
            logger.info(f"Report saved: {report.url}")
        else:
            logger.info("Skipping report creation as per configuration.")

        return final_vs_artifact_path

    except Exception as e:
        logger.error(f"Combined vectorstore and report pipeline failed: {e}", exc_info=True)
        if run:
            run.finish(exit_code=1)
        raise
    finally:
        if run:
            run.finish()
            logger.info(f"W&B Run finished: {run.url}") 