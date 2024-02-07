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
from typing import Any, List, Sequence, Callable

import tiktoken
import wandb
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document, BaseDocumentTransformer

from wandbot.utils import (
    get_logger,
    FastTextLangDetect,
    make_document_tokenization_safe,
    filter_smaller_documents,
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


class MarkdownTextTransformer(BaseDocumentTransformer):
    def __init__(self, lang_detect, chunk_size: int = 512):
        self.fasttext_model = lang_detect
        self.chunk_size: int = chunk_size
        self.length_function: Callable[[str], int]
        self.recursive_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            keep_separator=True,
            length_function=length_function,
        )
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
                ("####", "header_4"),
                ("#####", "header_5"),
                ("######", "header_6"),
            ]
        )

    def split_document_on_headers(
        self,
        document: Document,
    ) -> List[Document]:
        output_documents = []
        splits = self.header_splitter.split_text(document.page_content)
        for split in list(splits):
            output_documents.append(
                Document(
                    page_content=split.page_content, metadata=document.metadata
                )
            )
        return output_documents

    def recursively_merge_chunks(
        self,
        chunks: List[Document],
    ) -> List[Document]:
        if not chunks:  # check if chunks is empty
            return []  # return an empty list if chunks is empty
        merged_chunks = []
        current_chunk = chunks[0]
        current_length = length_function(current_chunk.page_content)
        for chunk in chunks[1:]:
            chunk_length = length_function(chunk.page_content)
            if current_length + chunk_length <= self.chunk_size:
                current_chunk.page_content += (
                    "\n\n" + chunk.page_content + "\n\n"
                )
                current_length += chunk_length
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk
                current_length = chunk_length
        merged_chunks.append(current_chunk)
        return merged_chunks

    def identify_document_language(self, document: Document) -> str:
        return self.fasttext_model.detect_language(document.page_content)

    def split_markdown_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        chunked_documents = []
        for document in documents:
            document_splits = self.split_document_on_headers(
                document=document,
            )
            split_chunks = self.recursive_splitter.split_documents(
                document_splits
            )
            merged_chunks = self.recursively_merge_chunks(
                split_chunks,
            )
            chunked_documents.extend(merged_chunks)

        for document in chunked_documents[:]:
            document.metadata["has_code"] = "```" in document.page_content
            document.metadata["language"] = self.identify_document_language(
                document
            )
        return chunked_documents

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        split_documents = self.split_markdown_documents(list(documents))
        transformed_documents = []
        for document in split_documents:
            transformed_documents.append(document)
        return transformed_documents


class CodeTextTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        chunk_size: int = 512,
    ):
        self.chunk_size: int = chunk_size
        self.length_function: Callable[[str], int]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split incoming code and return chunks using the AST."""
        chunked_documents = []
        for document in documents:
            file_extension = document.metadata.get("file_type", "")
            if file_extension in [".py", ".js", ".ts"]:
                language = {
                    ".py": Language.PYTHON,
                    ".js": Language.JS,
                    ".ts": Language.JS,
                }[file_extension]
                recursive_splitter = (
                    RecursiveCharacterTextSplitter.from_language(
                        language=language,
                        chunk_size=self.chunk_size,
                        chunk_overlap=0,
                        keep_separator=True,
                        length_function=length_function,
                    )
                )
                chunked_documents.extend(
                    recursive_splitter.split_documents([document])
                )
            elif file_extension in [".md", ".ipynb"]:
                chunked_documents.extend(
                    MarkdownTextTransformer().transform_documents([document])
                )
            else:
                chunked_documents.extend(
                    TokenTextSplitter().split_documents([document])
                )
        return chunked_documents

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        document_splits = []
        for document in list(documents):
            document_splits.extend(self.split_documents([document]))

        transformed_documents = []

        for document in list(document_splits):
            document.metadata["has_code"] = True
            document.metadata["language"] = "en"
            transformed_documents.append(document)

        return transformed_documents


class DocumentTransformer(BaseDocumentTransformer):
    def __init__(self, lang_detect, max_size: int = 512, min_size: int = 5):
        self.lang_detect = lang_detect
        self.chunk_size = max_size
        self.min_size = min_size
        self.markdown_transformer = MarkdownTextTransformer(
            lang_detect=lang_detect, chunk_size=self.chunk_size
        )
        self.code_transformer = CodeTextTransformer(chunk_size=self.chunk_size)

    def filter_smaller_documents(
        self, documents: List[Document], min_size: int = 5
    ) -> List[Document]:
        """Filters out nodes that are smaller than the chunk size.

        Args:
            text_nodes: A list of nodes.
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
            with transformed_file.open("w") as f:
                for document in transformed_documents:
                    f.write(json.dumps(document.dict()) + "\n")

            config["chunk_size"] = 512
            with open(transformed_file.parent / "config.json", "w") as f:
                json.dump(config, f)

            metadata["num_transformed_documents"] = len(transformed_documents)
            with open(transformed_file.parent / "metadata.json", "w") as f:
                json.dump(metadata, f)

            result_artifact.add_dir(
                str(transformed_file.parent),
                name=document_file.parent.name,
            )

    run.log_artifact(result_artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"
