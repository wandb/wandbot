import html
import json
import pathlib
import re
from hashlib import md5
from typing import Any, Callable, Dict, List, Sequence

import frontmatter
import markdown
import markdownify
from bs4 import BeautifulSoup, Comment
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import BaseDocumentTransformer, Document


def create_id_from_document(document: Document) -> str:
    contents = document.page_content + json.dumps(document.metadata)
    checksum = md5(contents.encode("utf-8")).hexdigest()
    return checksum


def convert_contents_to_soup(contents: str) -> BeautifulSoup:
    """Converts contents to BeautifulSoup object.

    Args:
        contents: The contents to convert.

    Returns:
        The BeautifulSoup object.
    """
    markdown_document = markdown.markdown(
        contents,
        extensions=[
            "toc",
            "pymdownx.extra",
            "pymdownx.blocks.admonition",
            "pymdownx.magiclink",
            "pymdownx.blocks.tab",
            "pymdownx.pathconverter",
            "pymdownx.saneheaders",
            "pymdownx.striphtml",
        ],
    )
    soup = BeautifulSoup(markdown_document, "html.parser")
    return soup


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Cleans the BeautifulSoup object.

    Args:
        soup: The BeautifulSoup object to clean.

    Returns:
        The cleaned BeautifulSoup object.
    """
    for img_tag in soup.find_all("img", src=True):
        img_tag.extract()
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    for p_tag in soup.find_all("p"):
        if not p_tag.text.strip():
            p_tag.decompose()
    return soup


def clean_contents(contents: str) -> str:
    """Cleans the contents.

    Args:
        contents: The contents to clean.

    Returns:
        The cleaned contents.
    """
    soup = convert_contents_to_soup(contents)
    soup = clean_soup(soup)
    cleaned_document = markdownify.MarkdownConverter(
        heading_style="ATX"
    ).convert_soup(soup)
    # Regular expression pattern to match import lines
    js_import_pattern = r"import .* from [‘’']@theme/.*[‘’'];\s*\n*"
    cleaned_document = re.sub(js_import_pattern, "", cleaned_document)
    cleaned_document = cleaned_document.replace("![]()", "\n")
    cleaned_document = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cleaned_document)
    cleaned_document = re.sub(r"\n{3,}", "\n\n", cleaned_document)
    cleaned_document = frontmatter.loads(cleaned_document).content
    return cleaned_document


def extract_frontmatter(file_path: pathlib.Path) -> Dict[str, Any]:
    """Extracts the frontmatter from a file.

    Args:
        file_path: The path to the file.

    Returns:
        The extracted frontmatter.
    """
    with open(file_path, "r") as f:
        contents = frontmatter.load(f)
        return {k: contents[k] for k in contents.keys()}


def strip_markdown_content(file_content):
    soup = convert_contents_to_soup(file_content)

    # Format headers with custom style and ensure they are not successive
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        header_text = header.get_text()
        formatted_header = f"\n\n---\n\n{header_text}\n\n\n"
        header.replace_with(formatted_header)

    # Replace <br> tags with newline characters
    for br in soup.find_all("br"):
        br.replace_with("\n\n\n")

    # Append a newline after each paragraph
    for p in soup.find_all("p"):
        p.append("\n\n\n")

    # Handle multiline code blocks enclosed in <pre> tags
    for pre in soup.find_all("pre"):
        code_text = pre.get_text()
        cleaned_code_text = code_text.strip("\n")
        # Ensure the code block is separated by newlines and enclosed in triple backticks
        formatted_code = f"\n\n\n```\n{cleaned_code_text}\n\n```\n\n\n"
        pre.replace_with(formatted_code)

    # Handle inline code blocks
    for code in soup.find_all("code"):
        if (
            code.parent.name != "pre"
        ):  # This checks if the <code> tag is not inside a <pre> tag
            inline_code_text = code.get_text()
            formatted_inline_code = f"`{inline_code_text}`"
            code.replace_with(formatted_inline_code)

    # Extract and unescape the HTML to plain text
    text = soup.get_text()
    unescaped_text = html.unescape(text)

    # Clean up escaped underscores and backticks
    clean_text = re.sub(r"\\_", "_", unescaped_text)
    clean_text = re.sub(r"\\`", "`", clean_text)

    # # Normalize double newlines to newlines
    clean_text = re.sub(r"\n\n", "\n", clean_text)

    # # Normalize triple or more newlines to double newlines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    return clean_text


def clean_markdown_chunk(chunk):
    chunk = chunk.replace("---", "\n\n")
    chunk = chunk.replace("```", "\n\n")
    chunk = re.sub(r"\n{3,}", "\n\n", chunk)
    chunk = chunk.strip()
    return chunk


class MarkdownTextTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        lang_detect,
        chunk_size: int = 768,
        length_function: Callable[[str], int] = None,
    ):
        self.fasttext_model = lang_detect
        self.chunk_size: int = chunk_size
        self.length_function: Callable[[str], int] = (
            length_function if length_function is not None else len
        )
        self.parent_recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n---\n",
                "\n```\n",
                "\n\n",
                "\n",
                "\s*",
            ],
            is_separator_regex=True,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size // 4,
            keep_separator=False,
            length_function=self.length_function,
        )

        self.child_recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n---\n",
                "\n```\n",
                "\n\n",
                "\n",
                "\s*",
            ],
            is_separator_regex=True,
            chunk_size=self.chunk_size // 2,
            chunk_overlap=self.chunk_size // 8,
            keep_separator=False,
            length_function=self.length_function,
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
            doc = Document(
                page_content=strip_markdown_content(document.page_content),
                metadata=document.metadata.copy(),
            )
            document_splits = self.parent_recursive_splitter.split_documents(
                documents=[doc],
            )
            for split in document_splits:
                chunk = Document(
                    page_content=clean_markdown_chunk(split.page_content),
                    metadata=split.metadata.copy(),
                )
                chunk.metadata["parent_id"] = create_id_from_document(chunk)
                chunk.metadata["has_code"] = "```" in chunk.page_content
                chunk.metadata["language"] = self.identify_document_language(
                    chunk
                )
                chunk.metadata["source_content"] = chunk.page_content
                chunked_documents.append(chunk)

            split_chunks = self.child_recursive_splitter.split_documents(
                chunked_documents
            )

            for chunk in split_chunks:
                chunk = Document(
                    page_content=clean_markdown_chunk(chunk.page_content),
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
