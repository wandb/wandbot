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

from typing import Any, Iterable, List, Optional, Union

import regex as re
import tiktoken
from langchain.schema import Document as LcDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import Document as LlamaDocument
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import CodeSplitter, TextSplitter
from tree_sitter_languages import get_parser
from wandbot.utils import get_logger

logger = get_logger(__name__)


def make_texts_tokenization_safe(documents: List[str]) -> List[str]:
    """Removes special tokens from the given documents.

    Args:
        documents: A list of strings representing the documents.

    Returns:
        A list of cleaned documents with special tokens removed.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    special_tokens_set = encoding.special_tokens_set

    def remove_special_tokens(text: str) -> str:
        """Removes special tokens from the given text.

        Args:
            text: A string representing the text.

        Returns:
            The text with special tokens removed.
        """
        for token in special_tokens_set:
            text = text.replace(token, "")
        return text

    cleaned_documents = []
    for document in documents:
        cleaned_document = remove_special_tokens(document)
        cleaned_documents.append(cleaned_document)
    return cleaned_documents


def non_whitespace_len(s: Union[str, bytes]) -> int:
    """Returns the length of the given string without whitespace.

    Args:
        s: A string.

    Returns:
        The length of the string without whitespace.
    """
    if isinstance(s, str):
        return len(re.sub("\s", "", s))
    return len(re.sub("\s", "", s.decode("utf-8")))


def split_by_headers(tree: Any) -> List[List[Any]]:
    """Splits the tree into chunks based on headers.

    Args:
        tree: The tree to split.

    Returns:
        A list of chunks where each chunk is a list of nodes.
    """
    chunks = []
    current_chunk = []
    for child in tree.root_node.children:
        if child.type == "atx_heading" or child.type == "setext_heading":
            if current_chunk:  # if current_chunk is not empty, add it to chunks
                chunks.append(current_chunk)
                current_chunk = []  # start a new chunk
            current_chunk.append(child)
        else:
            current_chunk.append(child)
    if current_chunk:  # add the last chunk if it's not empty
        chunks.append(current_chunk)
    return chunks


def split_large_chunks_by_code_blocks(
    chunks: List[List[Any]], max_chars: int
) -> List[List[Any]]:
    """Splits large chunks into smaller chunks based on code blocks.

    Args:
        chunks: A list of chunks where each chunk is a list of nodes.
        max_chars: The maximum number of characters allowed in a chunk.

    Returns:
        A list of smaller chunks where each chunk is a list of nodes.
    """
    new_chunks = []
    for chunk in chunks:
        current_chunk = []
        current_chars = 0
        for child in chunk:
            child_chars = non_whitespace_len(child.text)
            current_chars += child_chars
            current_chunk.append(child)
            if child.type == "fenced_code_block" and current_chars > max_chars:
                if (
                    current_chunk
                ):  # if current_chunk is not empty, add it to new_chunks
                    new_chunks.append(current_chunk)
                    current_chunk = []  # start a new chunk
                current_chars = 0
        if current_chunk:  # add the last chunk if it's not empty
            new_chunks.append(current_chunk)
    return new_chunks


def get_heading_level(chunk: List[Any]) -> Optional[int]:
    """Returns the heading level of the given chunk.

    Args:
        chunk: A list of nodes representing a chunk.

    Returns:
        The heading level of the chunk.
    """
    for child in chunk:
        if child.type == "atx_heading" or child.type == "setext_heading":
            for grandchild in child.children:
                if grandchild.type.startswith(
                    "atx"
                ) and grandchild.type.endswith("marker"):
                    return len(grandchild.text)
    return None


def merge_small_chunks(
    chunks: List[List[Any]], max_chars: int
) -> List[List[Any]]:
    """Merges small chunks into larger chunks based on maximum characters.

    Args:
        chunks: A list of chunks where each chunk is a list of nodes.
        max_chars: The maximum number of characters allowed in a chunk.

    Returns:
        A list of merged chunks where each chunk is a list of nodes.
    """
    merged_chunks = []
    current_chunk = []
    current_chars = 0
    for chunk in chunks:
        chunk_chars = sum(non_whitespace_len(child.text) for child in chunk)
        current_heading_level, chunk_heading_level = get_heading_level(
            current_chunk
        ), get_heading_level(chunk)
        cond = (
            current_heading_level is None and chunk_heading_level is None
        ) or (
            current_heading_level
            and chunk_heading_level
            and current_heading_level <= chunk_heading_level
        )
        if current_chars + chunk_chars <= max_chars and (
            not current_chunk or cond
        ):
            current_chunk.extend(chunk)
            current_chars += chunk_chars
        else:
            merged_chunks.append(current_chunk)
            current_chunk = chunk
            current_chars = chunk_chars
    if current_chunk:  # add the last chunk if it's not empty
        merged_chunks.append(current_chunk)
    return merged_chunks


def coalesce_small_chunks(
    chunks: List[List[Any]], min_chars: int = 100
) -> List[List[Any]]:
    """Coalesces small chunks into larger chunks based on minimum characters.

    Args:
        chunks: A list of chunks where each chunk is a list of nodes.
        min_chars: The minimum number of characters allowed in a chunk.

    Returns:
        A list of coalesced chunks where each chunk is a list of nodes.
    """
    coalesced_chunks = []
    i = 0
    while i < len(chunks):
        chunk_chars = sum(non_whitespace_len(child.text) for child in chunks[i])
        if chunk_chars < min_chars:  # if chunk is too small
            if i < len(chunks) - 1:  # if it's not the last chunk
                next_chunk_heading_level = get_heading_level(chunks[i + 1])
                current_chunk_heading_level = get_heading_level(chunks[i])
                if next_chunk_heading_level is None or (
                    current_chunk_heading_level is not None
                    and next_chunk_heading_level > current_chunk_heading_level
                ):
                    # if the next chunk is not a heading or is a heading of a higher level
                    chunks[i + 1] = (
                        chunks[i] + chunks[i + 1]
                    )  # prepend the chunk to the next chunk
                    i += 1  # skip to the next chunk
                    continue
            # if it's the last chunk or the next chunk is a heading of the same level
            if coalesced_chunks:  # if there are already some coalesced chunks
                coalesced_chunks[-1].extend(
                    chunks[i]
                )  # add the chunk to the previous chunk
            else:
                coalesced_chunks.append(chunks[i])
                i += 1
        else:
            coalesced_chunks.append(
                chunks[i]
            )  # add the chunk as a separate chunk
        i += 1
    return coalesced_chunks


def get_line_number(index: int, source_code: bytes) -> int:
    """Returns the line number corresponding to the given index in the source code.

    Args:
        index: The index in the source code.
        source_code: The source code as bytes.

    Returns:
        The line number corresponding to the index.
    """
    total_chars = 0
    for line_number, line in enumerate(
        source_code.splitlines(keepends=True), start=1
    ):
        total_chars += len(line)
        if total_chars > index:
            return line_number - 1
    return line_number


def coalesce_strings(strings: List[str], max_length: int) -> List[str]:
    """Coalesces strings into larger strings based on maximum length.

    Args:
        strings: A list of strings.
        max_length: The maximum length allowed for a coalesced string.

    Returns:
        A list of coalesced strings.
    """
    result = []
    current_string = ""

    for string in strings:
        if (
            non_whitespace_len(current_string) + non_whitespace_len(string)
            <= max_length
        ):
            current_string += "\n" + string
        else:
            result.append(current_string)
            current_string = string

    # Add the last remaining string
    if current_string:
        result.append(current_string)

    return result


def clean_extra_newlines(strings: List[str]) -> List[str]:
    """Cleans extra newlines in the given strings.

    Args:
        strings: A list of strings.

    Returns:
        A list of strings with extra newlines cleaned.
    """
    result = []
    for string in strings:
        string = re.sub(r"\n```\n", "CODEBREAK", string)
        string = re.sub(r"\n{2,}", "LINEBREAK", string)
        string = re.sub(r"\n", " ", string)
        string = re.sub(r"LINEBREAK", "\n\n", string)
        string = re.sub(r"CODEBREAK", "\n\n```\n\n", string)
        result.append(string)
    return result


class MarkdownSplitter(TextSplitter):
    """Splits text into chunks based on Markdown formatting.

    Attributes:
        sub_splitter: The sub-splitter used to split text into smaller chunks.
        chunk_size: The maximum size of each chunk.
    """

    sub_splitter: RecursiveCharacterTextSplitter = (
        RecursiveCharacterTextSplitter.from_language("markdown")
    )
    chunk_size: int = 1024

    def __init__(self, **kwargs):
        """Initializes the MarkdownSplitter instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.sub_splitter = RecursiveCharacterTextSplitter.from_language(
            "markdown"
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MarkdownSplitter"

    def _chunk_text(self, text: str) -> Iterable[str]:
        """Split text into chunks."""
        parser = get_parser("markdown")
        tree = parser.parse(text.encode("utf-8"))

        chunks: List[List[Any]] = split_by_headers(tree)
        chunks = split_large_chunks_by_code_blocks(chunks, self.chunk_size)
        chunks = merge_small_chunks(chunks, self.chunk_size)
        for chunk in chunks:
            if chunk:
                chunk_bytes = chunk[0].start_byte, chunk[-1].end_byte
                chunk_lines = text.encode("utf-8").splitlines()[
                    get_line_number(
                        chunk_bytes[0], text.encode("utf-8")
                    ) : get_line_number(chunk_bytes[1], text.encode("utf-8"))
                    + 1
                ]
                chunk_str = ""
                for line in chunk_lines:
                    if line.decode().endswith("\n"):
                        chunk_str += line.decode()
                    else:
                        chunk_str += line.decode() + "\n"

                for split in self.sub_splitter.split_documents(
                    [LcDocument(page_content=chunk_str)]
                ):
                    split_content = split.page_content

                    yield split_content

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = list(self._chunk_text(text))
        chunks = coalesce_strings(chunks, self.chunk_size)
        chunks = clean_extra_newlines(chunks)
        chunks = make_texts_tokenization_safe(chunks)
        return chunks


class CustomCodeSplitter(CodeSplitter):
    """Splits text into chunks based on custom code formatting.

    Attributes:
        language: The programming language of the code.
        max_chars: The maximum number of characters allowed in a chunk.
    """

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        text_splits = super().split_text(text)
        text_splits = make_texts_tokenization_safe(text_splits)
        return text_splits


def convert_lc_to_llama(document: LcDocument) -> LlamaDocument:
    """Converts a Langchain document to a Llama document.

    Args:
        document: A Langchain document.

    Returns:
        A Llama document.
    """
    return LlamaDocument.from_langchain_format(document)


def load(documents: List[LcDocument], chunk_size: int = 1024) -> List[Any]:
    """Loads documents and returns a list of nodes.

    Args:
        documents: A list of documents.
        chunk_size: The size of each chunk.

    Returns:
        A list of nodes.
    """
    md_parser: SimpleNodeParser = SimpleNodeParser(
        text_splitter=MarkdownSplitter(chunk_size=chunk_size)
    )
    code_parser: SimpleNodeParser = SimpleNodeParser(
        text_splitter=CustomCodeSplitter(
            language="python", max_chars=chunk_size
        )
    )

    llama_docs: List[LlamaDocument] = list(
        map(lambda x: convert_lc_to_llama(x), documents)
    )

    nodes: List[Any] = []
    for doc in llama_docs:
        if doc.metadata["file_type"] == ".py":
            parser = code_parser
        else:
            parser = md_parser
        nodes.extend(parser.get_nodes_from_documents([doc]))
    return nodes
