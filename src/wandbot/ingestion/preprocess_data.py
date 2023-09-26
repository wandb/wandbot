import regex as re
import tiktoken
from langchain.schema import Document as LcDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import Document as LlamaDocument
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import CodeSplitter, TextSplitter
from tree_sitter_languages import get_parser
from typing import Iterable, List
from wandbot.utils import get_logger

logger = get_logger(__name__)


def make_texts_tokenization_safe(documents):
    encoding = tiktoken.get_encoding("cl100k_base")
    special_tokens_set = encoding.special_tokens_set

    def remove_special_tokens(text):
        for token in special_tokens_set:
            text = text.replace(token, "")
        return text

    cleaned_documents = []
    for document in documents:
        cleaned_document = remove_special_tokens(document)
        cleaned_documents.append(cleaned_document)
    return cleaned_documents


def non_whitespace_len(s) -> int:
    if isinstance(s, str):
        return len(re.sub("\s", "", s))
    return len(re.sub("\s", "", s.decode("utf-8")))


def split_by_headers(tree):
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


def split_large_chunks_by_code_blocks(chunks, max_chars):
    new_chunks = []
    for chunk in chunks:
        current_chunk = []
        current_chars = 0
        for child in chunk:
            child_chars = non_whitespace_len(child.text)
            current_chars += child_chars
            current_chunk.append(child)
            if child.type == "fenced_code_block" and current_chars > max_chars:
                if current_chunk:  # if current_chunk is not empty, add it to new_chunks
                    new_chunks.append(current_chunk)
                    current_chunk = []  # start a new chunk
                current_chars = 0
        if current_chunk:  # add the last chunk if it's not empty
            new_chunks.append(current_chunk)
    return new_chunks


def get_heading_level(chunk):
    for child in chunk:
        if child.type == "atx_heading" or child.type == "setext_heading":
            for grandchild in child.children:
                if grandchild.type.startswith("atx") and grandchild.type.endswith("marker"):
                    return len(grandchild.text)
    return None


def merge_small_chunks(chunks, max_chars):
    merged_chunks = []
    current_chunk = []
    current_chars = 0
    for chunk in chunks:
        chunk_chars = sum(non_whitespace_len(child.text) for child in chunk)
        current_heading_level, chunk_heading_level = get_heading_level(current_chunk), get_heading_level(chunk)
        cond = (current_heading_level is None and chunk_heading_level is None) or (
            current_heading_level and chunk_heading_level and current_heading_level <= chunk_heading_level
        )
        if current_chars + chunk_chars <= max_chars and (not current_chunk or cond):
            current_chunk.extend(chunk)
            current_chars += chunk_chars
        else:
            merged_chunks.append(current_chunk)
            current_chunk = chunk
            current_chars = chunk_chars
    if current_chunk:  # add the last chunk if it's not empty
        merged_chunks.append(current_chunk)
    return merged_chunks


def coalesce_small_chunks(chunks, min_chars=100):
    coalesced_chunks = []
    i = 0
    while i < len(chunks):
        chunk_chars = sum(non_whitespace_len(child.text) for child in chunks[i])
        if chunk_chars < min_chars:  # if chunk is too small
            if i < len(chunks) - 1:  # if it's not the last chunk
                next_chunk_heading_level = get_heading_level(chunks[i + 1])
                current_chunk_heading_level = get_heading_level(chunks[i])
                if next_chunk_heading_level is None or (
                    current_chunk_heading_level is not None and next_chunk_heading_level > current_chunk_heading_level
                ):
                    # if the next chunk is not a heading or is a heading of a higher level
                    chunks[i + 1] = chunks[i] + chunks[i + 1]  # prepend the chunk to the next chunk
                    i += 1  # skip to the next chunk
                    continue
            # if it's the last chunk or the next chunk is a heading of the same level
            if coalesced_chunks:  # if there are already some coalesced chunks
                coalesced_chunks[-1].extend(chunks[i])  # add the chunk to the previous chunk
            else:
                coalesced_chunks.append(chunks[i])
                i += 1
        else:
            coalesced_chunks.append(chunks[i])  # add the chunk as a separate chunk
        i += 1
    return coalesced_chunks


def get_line_number(index: int, source_code: bytes) -> int:
    total_chars = 0
    for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
        total_chars += len(line)
        if total_chars > index:
            return line_number - 1
    return line_number


def coalesce_strings(strings, max_length):
    result = []
    current_string = ""

    for string in strings:
        if non_whitespace_len(current_string) + non_whitespace_len(string) <= max_length:
            current_string += "\n" + string
        else:
            result.append(current_string)
            current_string = string

    # Add the last remaining string
    if current_string:
        result.append(current_string)

    return result


def clean_extra_newlines(strings):
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
    sub_splitter = RecursiveCharacterTextSplitter.from_language("markdown")
    chunk_size = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sub_splitter = RecursiveCharacterTextSplitter.from_language("markdown")

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MarkdownSplitter"

    def _chunk_text(self, text: str) -> Iterable[str]:
        """Split text into chunks."""
        parser = get_parser("markdown")
        tree = parser.parse(text.encode("utf-8"))

        chunks = split_by_headers(tree)
        chunks = split_large_chunks_by_code_blocks(chunks, self.chunk_size)
        chunks = merge_small_chunks(chunks, self.chunk_size)
        for chunk in chunks:
            if chunk:
                chunk_bytes = chunk[0].start_byte, chunk[-1].end_byte
                chunk_lines = text.encode("utf-8").splitlines()[
                    get_line_number(chunk_bytes[0], text.encode("utf-8")) : get_line_number(
                        chunk_bytes[1], text.encode("utf-8")
                    )
                    + 1
                ]
                chunk_str = ""
                for line in chunk_lines:
                    if line.decode().endswith("\n"):
                        chunk_str += line.decode()
                    else:
                        chunk_str += line.decode() + "\n"

                for split in self.sub_splitter.split_documents([LcDocument(page_content=chunk_str)]):
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
    def split_text(self, text: str) -> List[str]:
        text_splits = super().split_text(text)
        text_splits = make_texts_tokenization_safe(text_splits)
        return text_splits


def convert_lc_to_llama(document: LcDocument):
    return LlamaDocument.from_langchain_format(document)


def load(documents, chunk_size=1024):
    md_parser = SimpleNodeParser(text_splitter=MarkdownSplitter(chunk_size=chunk_size))
    code_parser = SimpleNodeParser(text_splitter=CustomCodeSplitter(language="python", max_chars=chunk_size))

    llama_docs = list(map(lambda x: convert_lc_to_llama(x), documents))

    nodes = []
    for doc in llama_docs:
        if doc.metadata["file_type"] == ".py":
            parser = code_parser
        else:
            parser = md_parser
        nodes.extend(node_parser.get_nodes_from_documents([doc]))
    return nodes
