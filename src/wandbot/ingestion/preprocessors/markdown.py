import json
from hashlib import md5
from typing import Any, Callable, Dict, List, Optional, Sequence, TypedDict

from langchain.text_splitter import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import BaseDocumentTransformer, Document

from wandbot.utils import FastTextLangDetect, FasttextModelConfig


class LineType(TypedDict):
    """Line type as typed dict."""

    metadata: Dict[str, str]
    content: str


class HeaderType(TypedDict):
    """Header type as typed dict."""

    level: int
    name: str
    data: str


def create_id_from_document(document: Document) -> str:
    contents = document.page_content + json.dumps(document.metadata)
    checksum = md5(contents.encode("utf-8")).hexdigest()
    return checksum


def prefix_headers_based_on_metadata(chunk):
    # Headers ordered by markdown header levels
    markdown_header_prefixes = ["#", "##", "###", "####", "#####", "######"]
    markdown_header_prefixes_map = {
        f"header_{i}": prefix
        for i, prefix in enumerate(markdown_header_prefixes)
    }

    # Generate headers from metadata
    headers_from_metadata = [
        f"{markdown_header_prefixes_map[level]} {title}"
        for level, title in chunk["metadata"].items()
    ]

    # Join the generated headers with new lines
    headers_str = "\n".join(headers_from_metadata) + "\n"

    # Check if the page_content starts with a header
    if chunk["content"].lstrip().startswith(tuple(markdown_header_prefixes)):
        # Find the first newline to locate the end of the existing header
        first_newline_index = chunk["content"].find("\n")
        if first_newline_index != -1:
            # Remove the existing header and prefix with generated headers
            modified_content = (
                headers_str + chunk["content"][first_newline_index + 1 :]
            )
        else:
            # If there's no newline, the entire content is a header, replace it
            modified_content = headers_str
    else:
        # If it doesn't start with a header, just prefix with generated headers
        modified_content = headers_str + chunk["content"]

    return {"metadata": chunk["metadata"], "content": modified_content}


class CustomMarkdownTextSplitter(MarkdownHeaderTextSplitter):
    """Splitting markdown files based on specified headers."""

    def __init__(self, chunk_size: Optional[int] = None, **kwargs):
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
            ("#####", "header_5"),
            ("######", "header_6"),
        ]
        self.max_length = chunk_size
        super().__init__(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False,
            strip_headers=False,
        )

    def aggregate_lines_to_chunks(
        self, lines: List[LineType]
    ) -> List[Document]:
        aggregated_chunks: List[LineType] = []

        for line in lines:
            should_append = True

            # Attempt to aggregate with an earlier chunk if possible
            for i in range(len(aggregated_chunks) - 1, -1, -1):
                previous_chunk = aggregated_chunks[i]
                # Check if the current line's metadata is a child or same level of the previous chunk's metadata
                if all(
                    item in line["metadata"].items()
                    for item in previous_chunk["metadata"].items()
                ):
                    potential_new_content = (
                        previous_chunk["content"] + "  \n\n" + line["content"]
                    )
                    if (
                        self.max_length is None
                        or len(potential_new_content) <= self.max_length
                    ):
                        # If adding the current line does not exceed chunk_size, merge it into the previous chunk
                        aggregated_chunks[i]["content"] = potential_new_content
                        should_append = False
                        break
                    else:
                        # If it exceeds chunk_size, no further checks are needed, break to append as a new chunk
                        break

            if should_append:
                # Append as a new chunk if it wasn't merged into an earlier one
                aggregated_chunks.append(line)

        # Prefix headers based on metadata
        aggregated_chunks = [
            prefix_headers_based_on_metadata(chunk)
            for chunk in aggregated_chunks
        ]
        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into smaller documents.

        Args:
            documents: A list of documents.

        Returns:
            A list of documents.
        """
        split_documents = []
        for document in documents:
            for chunk in self.split_text(document.page_content):
                split_documents.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata=document.metadata,
                    )
                )
        return split_documents


class MarkdownTextTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        lang_detect,
        chunk_size: int = 512,
        length_function: Callable[[str], int] = None,
    ):
        self.fasttext_model = lang_detect
        self.chunk_size: int = chunk_size
        self.length_function: Callable[[str], int] = (
            length_function if length_function is not None else len
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            keep_separator=True,
            length_function=self.length_function,
        )
        self.header_splitter = CustomMarkdownTextSplitter(
            chunk_size=self.chunk_size * 2,
        )

    def identify_document_language(self, document: Document) -> str:
        if "language" in document.metadata:
            return document.metadata["language"]
        else:
            return self.fasttext_model.detect_language(document.page_content)

    def split_markdown_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        final_chunks = []
        chunked_documents = []
        for document in documents:
            document_splits = self.header_splitter.split_documents(
                documents=[document],
            )
            for split in document_splits:
                chunk = Document(
                    page_content=split.page_content,
                    metadata=split.metadata.copy(),
                )
                chunk.metadata["parent_id"] = create_id_from_document(chunk)
                chunk.metadata["has_code"] = "```" in chunk.page_content
                chunk.metadata["language"] = self.identify_document_language(
                    chunk
                )
                chunk.metadata["source_content"] = chunk.page_content
                chunked_documents.append(chunk)

            split_chunks = self.recursive_splitter.split_documents(
                chunked_documents
            )

            for chunk in split_chunks:
                chunk = Document(
                    page_content=chunk.page_content,
                    metadata=chunk.metadata.copy(),
                )
                chunk.metadata["id"] = create_id_from_document(chunk)
                final_chunks.append(chunk)

        return final_chunks

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        split_documents = self.split_markdown_documents(list(documents))
        transformed_documents = []
        for document in split_documents:
            transformed_documents.append(document)
        return transformed_documents


if __name__ == "__main__":
    lang_detect = FastTextLangDetect(
        FasttextModelConfig(
            fasttext_file_path="/media/mugan/data/wandb/projects/wandbot/data/cache/models/lid.176.bin"
        )
    )

    data_file = open(
        "/media/mugan/data/wandb/projects/wandbot/data/cache/raw_data/docodile_store/docodile_en/documents.jsonl"
    ).readlines()
    source_document = json.loads(data_file[0])

    source_document = Document(**source_document)

    markdown_transformer = MarkdownTextTransformer(
        lang_detect=lang_detect, chunk_size=768 // 2
    )

    transformed_documents = markdown_transformer.transform_documents(
        [source_document]
    )

    for document in transformed_documents:
        print(document.page_content)
        print(json.dumps(document.metadata, indent=2))
        print("*" * 80)
