"""This module contains classes and functions for preprocessing data in the Wandbot ingestion system.

The module includes the following classes:
- `MarkdownSplitter`: A class for splitting text into chunks based on Markdown formatting.
- `CustomCodeSplitter`: A class for splitting text into chunks based on custom code formatting.

The module also includes the following functions:
- `make_texts_tokenization_safe`: Removes special tokens from the given documents.
- `non_whitespace_len`: Returns the length of the given string without whitespace.
- `split_by_headers`: Splits the tree into chunks based on headers.
- `split_large_chunks_by_code_blocks`: Splits large chunks into smaller chunks based on code blocks.
- `get_line_number`: Returns the line number of a given index in the source code.
- `convert_lc_to_llama`: Converts a Langchain document to a Llama document.
- `load`: Loads documents and returns a list of nodes.

Typical usage example:

    documents = [document1, document2, document3]
    nodes = load(documents, chunk_size=1024)
"""

import json
import pathlib
from typing import Any, List, Sequence

import tiktoken
from langchain_core.documents import BaseDocumentTransformer

import wandb
from wandbot.ingestion.preprocessors.markdown import MarkdownTextTransformer
from wandbot.ingestion.preprocessors.source_code import CodeTextTransformer
from wandbot.schema.document import Document
from wandbot.utils import (
    FastTextLangDetect,
    filter_smaller_documents,
    get_logger,
    make_document_tokenization_safe,
)

logger = get_logger(__name__)


class Tokenizer:
    def __init__(self, model_name):
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


tokenizer = Tokenizer("gpt2")


def length_function(content: str) -> int:
    return len(tokenizer.encode(content))


def len_function_with_doc(document: Document) -> int:
    return len(tokenizer.encode(document.page_content))


class DocumentTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        lang_detect,
        chunk_size: int,
        chunk_multiplier: int,
        chunk_overlap: int,
        min_size: int = 5,
        length_function=None,
    ):
        self.lang_detect = lang_detect
        self.chunk_size = chunk_size
        self.chunk_multiplier = chunk_multiplier
        self.chunk_overlap = chunk_overlap
        self.min_size = min_size
        self.length_function = length_function
        self.markdown_transformer = MarkdownTextTransformer(
            lang_detect=lang_detect,
            chunk_size=self.chunk_size,
            chunk_multiplier=self.chunk_multiplier,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )
        self.code_transformer = CodeTextTransformer(
            lang_detect=self.lang_detect,
            chunk_size=self.chunk_size,
            chunk_multiplier=self.chunk_multiplier,
            chunk_overlap=self.chunk_overlap,
            length_function=length_function,
        )

    def filter_smaller_documents(
        self, documents: List[Document], min_size: int = 5
    ) -> List[Document]:
        """Filters out nodes that are smaller than the chunk size.

        Args:
            documents: A list of nodes.
            min_size: The minimum size of a node.
        Returns:
            A list of nodes.
        """

        for node in documents:
            content = node.page_content
            if length_function(content) >= min_size:
                yield node

    def standardize_metadata(self, documents: List[Document]) -> List[Document]:
        for document in documents:
            metadata = document.metadata
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    if isinstance(value, list):
                        metadata[key] = " ".join(value)
                    elif isinstance(value, dict):
                        metadata[key] = json.dumps(value)
                    elif isinstance(value, (set, tuple)):
                        metadata[key] = " ".join(list(value))
                    else:
                        metadata[key] = None
            metadata = {k: v for k, v in metadata.items() if v is not None}
            document.metadata = metadata
        return documents

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        transformed_documents = []
        for document in list(documents):
            document = make_document_tokenization_safe(document)
            if document.metadata.get("source_type", "") == "code":
                transformed_documents.extend(
                    self.code_transformer.transform_documents([document])
                )
            else:
                transformed_documents.extend(
                    self.markdown_transformer.transform_documents([document])
                )
        transformed_documents = list(
            self.filter_smaller_documents(
                transformed_documents, min_size=self.min_size
            )
        )
        transformed_documents = list(
            self.standardize_metadata(transformed_documents)
        )

        transformed_documents = filter_smaller_documents(
            transformed_documents, min_size=self.min_size
        )
        return transformed_documents

    def transform_document(self, document: Document) -> Document:
        return self.transform_documents([document])[0]


def process_document_file(
    documents: List[Document], transformer: DocumentTransformer
) -> List[Document]:
    transformed_documents = transformer.transform_documents(documents)

    return list(transformed_documents)


def run_preprocessing_pipeline(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = "transformed_data",
    debug: bool = False,
) -> str:
    run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
        project=project, entity=entity, job_type="preprocess_data"
    )
    artifact: wandb.Artifact = run.use_artifact(
        source_artifact_path, type="dataset"
    )
    artifact_dir: pathlib.Path = pathlib.Path(artifact.download())

    lang_detect = FastTextLangDetect()
    result_artifact = wandb.Artifact(result_artifact_name, type="dataset")
    all_source_metadata = {} # Initialize dictionary to store metadata for all sources

    # Find all unique parent directories containing documents.jsonl
    source_directories = set()
    for doc_file in artifact_dir.rglob("documents.jsonl"):
        source_directories.add(doc_file.parent)

    if not source_directories:
        logger.warning(f"No directories with documents.jsonl found in artifact: {source_artifact_path}")
        run.finish()
        return f"{entity}/{project}/{result_artifact_name}:latest" # Return but artifact will be empty

    for source_dir in source_directories:
        logger.info(f"Processing source directory: {source_dir.name}")
        
        # 1. Read source-specific config
        config_path = source_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found for source: {config_path}"
            )
        with config_path.open() as f:
            config = json.load(f)
            if "chunk_size" not in config:
                raise ValueError(
                    f"'chunk_size' not found in config file: {config_path}"
                )
            if "chunk_multiplier" not in config:
                raise ValueError(
                    f"'chunk_multiplier' not found in config file: {config_path}"
                )
            chunk_size = config["chunk_size"]
            chunk_multiplier = config["chunk_multiplier"]

        # 2. Instantiate transformer with specific config
        transformer = DocumentTransformer(
            lang_detect=lang_detect,
            chunk_size=chunk_size,
            chunk_multiplier=chunk_multiplier,
            min_size=5, # Consider making min_size configurable too?
            length_function=length_function,
        )

        # 3. Load documents for this source
        document_file = source_dir / "documents.jsonl"
        with document_file.open() as f:
            documents = []
            for i, line in enumerate(f):
                if debug and i >= 3:
                    logger.warning(f"DEBUG MODE: Reading only first 3 documents from {document_file}")
                    break
                documents.append(Document(**json.loads(line)))
        logger.info(f"Loaded {len(documents)} initial file documents from source file: {document_file}{' (DEBUG MODE limit applied)' if debug else ''}")
        
        # 4. Transform documents
        # Replacing process_document_file call with direct transformation
        logger.info("Applying document transformers to generate text chunks...")
        transformed_documents = transformer.transform_documents(documents)
        transformed_documents = list(transformed_documents) # Ensure it's a list
        logger.info(f"Generated {len(transformed_documents)} text chunks.")

        # 5. Prepare output paths and save results
        metadata_path = source_dir / "metadata.json"
        if not metadata_path.exists():
             raise FileNotFoundError(
                f"Metadata file not found for source: {metadata_path}"
            )
        with metadata_path.open() as f:
            metadata = json.load(f)

        # Determine output directory relative to a base 'transformed_data' cache
        # Assuming config['data_source']['cache_dir'] points to the *raw* cache base
        # e.g., data/cache/raw_data
        raw_cache_base = pathlib.Path(config["data_source"]["cache_dir"]).parent
        transformed_data_cache_dir = raw_cache_base / "transformed_data"
        output_dir = transformed_data_cache_dir / source_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        transformed_file = output_dir / document_file.name
        output_config_path = output_dir / config_path.name
        output_metadata_path = output_dir / metadata_path.name

        # Write transformed documents
        with transformed_file.open("w") as of:
            logger.info(f"Writing {len(transformed_documents)} text chunks to {transformed_file}")
            for document in transformed_documents:
                of.write(json.dumps(document.dict()) + "\n")

        # Update and write config (already contains correct chunk settings from input)
        with output_config_path.open("w") as of:
            json.dump(config, of)

        # Update and write metadata
        keys_to_remove = [
            "vectordb_index_artifact_url",
            "vector_store_auth_token",
            "embeddings_query_input_type",
            "embeddings_document_input_type",
        ]
        for key in keys_to_remove:
            metadata.pop(key, None) # Use pop with default None to avoid KeyError if key doesn't exist

        metadata["num_transformed_documents"] = len(transformed_documents)
        with output_metadata_path.open("w") as of:
            logger.info(f"Writing updated metadata to {output_metadata_path}")
            json.dump(metadata, of)

        # Store this source's metadata
        all_source_metadata[source_dir.name] = metadata

        # 6. Add processed directory to the result artifact
        result_artifact.add_dir(str(output_dir), name=source_dir.name)

    # Prepare description string now that all_source_metadata is populated
    description_string = f"Preprocessed data artifact containing transformed text chunks for {len(all_source_metadata)} sources."
    if debug:
        description_string += " (DEBUG MODE: Processed only first source and first 3 documents)."
    description_string += "\nMetadata per source:\n"
    description_string += json.dumps(all_source_metadata, indent=2) # Format as JSON for description

    # Set metadata and description on the artifact object itself
    result_artifact.metadata = all_source_metadata
    result_artifact.description = description_string

    # Define intended aliases (tags will be added *after* logging)
    intended_aliases = [] # Default alias

    # Log artifact - only pass artifact object and aliases
    run.log_artifact(
        result_artifact,
        aliases=intended_aliases
    )
    logger.info(f"Artifact {result_artifact.name} logged with aliases: {intended_aliases}, now uploading...")


    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"