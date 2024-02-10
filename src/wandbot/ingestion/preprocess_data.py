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
import wandb
from langchain_core.documents import BaseDocumentTransformer, Document
from wandbot.ingestion.preprocessors.markdown import MarkdownTextTransformer
from wandbot.ingestion.preprocessors.source_code import CodeTextTransformer
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
        max_size: int = 512,
        min_size: int = 5,
        length_function=None,
    ):
        self.lang_detect = lang_detect
        self.chunk_size = max_size
        self.min_size = min_size
        self.length_function = length_function
        self.markdown_transformer = MarkdownTextTransformer(
            lang_detect=lang_detect,
            chunk_size=self.chunk_size,
            length_function=self.length_function,
        )
        self.code_transformer = CodeTextTransformer(
            lang_detect=self.lang_detect,
            chunk_size=self.chunk_size,
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


def load(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = "transformed_data",
) -> str:
    run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
        project=project, entity=entity, job_type="preprocess_data"
    )
    artifact: wandb.Artifact = run.use_artifact(
        source_artifact_path, type="dataset"
    )
    artifact_dir: str = artifact.download()

    document_files: List[pathlib.Path] = list(
        pathlib.Path(artifact_dir).rglob("documents.jsonl")
    )

    lang_detect = FastTextLangDetect()
    transformer = DocumentTransformer(
        lang_detect=lang_detect,
        max_size=512,
        min_size=5,
        length_function=length_function,
    )

    result_artifact = wandb.Artifact(result_artifact_name, type="dataset")

    for document_file in document_files:
        with document_file.open() as f:
            documents = [Document(**json.loads(line)) for line in f]
            transformed_documents = process_document_file(
                documents, transformer
            )
            config = json.load((document_file.parent / "config.json").open())
            metadata = json.load(
                (document_file.parent / "metadata.json").open()
            )
            cache_dir = (
                pathlib.Path(config["data_source"]["cache_dir"]).parent
                / "transformed_data"
            )

            transformed_file = (
                cache_dir / document_file.parent.name / document_file.name
            )

            transformed_file.parent.mkdir(parents=True, exist_ok=True)
            with transformed_file.open("w") as of:
                for document in transformed_documents:
                    of.write(json.dumps(document.dict()) + "\n")

            config["chunk_size"] = 512
            with open(transformed_file.parent / "config.json", "w") as of:
                json.dump(config, of)

            metadata["num_transformed_documents"] = len(transformed_documents)
            with open(transformed_file.parent / "metadata.json", "w") as of:
                json.dump(metadata, of)

            result_artifact.add_dir(
                str(transformed_file.parent),
                name=document_file.parent.name,
            )

    run.log_artifact(result_artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"
