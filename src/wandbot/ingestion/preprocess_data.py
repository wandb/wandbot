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

from typing import Any, List

import tiktoken
from langchain.schema import Document as LcDocument
from llama_index import Document as LlamaDocument
from llama_index.node_parser import CodeSplitter, MarkdownNodeParser

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
    md_parser = MarkdownNodeParser()
    code_parser = CustomCodeSplitter(language="python", max_chars=chunk_size)

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
