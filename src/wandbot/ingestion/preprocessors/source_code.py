import ast
import json
from typing import Any, Callable, List, Sequence

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import BaseDocumentTransformer, Document
from wandbot.ingestion.preprocessors.md import (
    MarkdownTextTransformer,
    create_id_from_document,
)
from wandbot.utils import FastTextLangDetect, FasttextModelConfig


def has_sufficient_content(file_content, min_line_count=10):
    """Check if the file has a minimum number of substantive lines."""
    lines = [
        line
        for line in file_content.split("\n")
        if line.strip() and not line.strip().startswith(("#", "//"))
    ]
    return len(lines) >= min_line_count


def remove_comments_and_docstrings(source):
    """Remove all top-level constructs except functions, classes, and async functions from the Python source code."""
    tree = ast.parse(source)
    new_body = []
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
        ):
            new_body.append(node)  # Keep function, class, and async definitions

    tree.body = new_body
    return ast.unparse(tree)


def load_code_file(file_content):
    if has_sufficient_content(file_content):
        cleaned_content = remove_comments_and_docstrings(file_content)
        return cleaned_content
    return None


class CodeTextTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        lang_detect,
        chunk_size: int = 1024,
        length_function: Callable[[str], int] = None,
    ):
        self.lang_detect = lang_detect
        self.chunk_size: int = chunk_size
        self.length_function: Callable[[str], int] = (
            length_function if length_function is not None else len
        )
        self.parent_python_code_splitter = RecursiveCharacterTextSplitter(
            separators=[
                # First, handle decorators
                "\n\s*@",
                # Then, handle definitions
                "\n\s*class\s+",
                "\n\s*def\s+",
                "\n\s*async\s+def\s+",
                # Split along control flow statements
                "\n\s*if\s+",
                "\n\s*else\s+",
                "\n\s*elif\s+",
                "\n\s*for\s+",
                "\n\s*async\s+for\s+",
                "\n\s*while\s+",
                "\n\s*async\s+while\s+",
                "\n\s*with\s+",
                "\n\s*async\s+with\s+",
                "\n\s*try\s+",
                "\n\s*except\s+",
                "\n\s*finally\s+",
                "\n\s*match\s+",
                "\n\s*case\s+",
                "\n\n",
                "\n",
                "\s*",
            ],
            is_separator_regex=True,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size // 4,
            keep_separator=True,
            length_function=self.length_function,
        )

        self.child_python_code_splitter = RecursiveCharacterTextSplitter(
            separators=[
                # First, handle decorators
                "\n\s*@",
                # Then, handle definitions
                "\n\s*class\s+",
                "\n\s*def\s+",
                "\n\s*async\s+def\s+",
                # Split along control flow statements
                "\n\s*if\s+",
                "\n\s*else\s+",
                "\n\s*elif\s+",
                "\n\s*for\s+",
                "\n\s*async\s+for\s+",
                "\n\s*while\s+",
                "\n\s*async\s+while\s+",
                "\n\s*with\s+",
                "\n\s*async\s+with\s+",
                "\n\s*try\s+",
                "\n\s*except\s+",
                "\n\s*finally\s+",
                "\n\s*match\s+",
                "\n\s*case\s+",
                "\n\n",
                "\n",
                "\s*",
            ],
            is_separator_regex=True,
            chunk_size=self.chunk_size // 2,
            chunk_overlap=self.chunk_size // 4,
            keep_separator=True,
            length_function=self.length_function,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split incoming code and return chunks using the AST."""

        # First chunk into parent documents

        chunked_documents = []
        for document in documents:
            file_extension = document.metadata.get("file_type", "")
            if file_extension == ".py":
                doc_content = load_code_file(document.page_content)
                if doc_content is not None:
                    document = Document(
                        page_content=doc_content,
                        metadata=document.metadata.copy(),
                    )
                    chunked_documents.extend(
                        self.parent_python_code_splitter.transform_documents(
                            [document]
                        )
                    )
            elif file_extension in [".md", ".ipynb"]:
                chunked_documents.extend(
                    MarkdownTextTransformer(
                        lang_detect=self.lang_detect,
                    ).transform_documents([document])
                )
            else:
                chunked_documents.extend(
                    TokenTextSplitter(
                        chunk_size=self.chunk_size
                    ).split_documents([document])
                )

        # make new documents from the chunks with updated metadata
        parent_documents = []
        for split in chunked_documents:
            document_content = split.page_content

            chunk = Document(
                page_content=document_content,
                metadata=split.metadata.copy(),
            )
            chunk.metadata["parent_id"] = create_id_from_document(chunk)
            chunk.metadata["has_code"] = True
            chunk.metadata["language"] = chunk.metadata.get("language", "en")
            chunk.metadata["source_content"] = split.metadata.get(
                "source_content", document_content
            )
            parent_documents.append(chunk)

        # now make children documents from the parent documents
        final_chunks = []
        for document in parent_documents:
            file_extension = document.metadata.get("file_type", "")
            if file_extension == ".py":
                chunks = self.child_python_code_splitter.transform_documents(
                    [document]
                )
                final_chunks.extend(chunks)

            elif file_extension in [".md", ".ipynb"]:
                final_chunks.extend(
                    MarkdownTextTransformer(
                        lang_detect=self.lang_detect,
                    ).transform_documents([document])
                )

            else:
                final_chunks.extend(
                    TokenTextSplitter(
                        chunk_size=self.chunk_size // 2,
                        chunk_overlap=self.chunk_size // 8,
                    ).split_documents([document])
                )

        # now add the ids for the final chunks
        output_chunks = []
        for chunk in final_chunks:
            chunk = Document(
                page_content=chunk.page_content,
                metadata=chunk.metadata.copy(),
            )
            chunk.metadata["id"] = create_id_from_document(chunk)
            output_chunks.append(chunk)

        return output_chunks

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        document_splits = []
        for document in list(documents):
            document_splits.extend(self.split_documents([document]))

        return document_splits


if __name__ == "__main__":
    lang_detect = FastTextLangDetect(
        FasttextModelConfig(
            fasttext_file_path="/media/mugan/data/wandb/projects/wandbot/data/cache/models/lid.176.bin"
        )
    )

    data_file = open(
        "/media/mugan/data/wandb/projects/wandbot/data/cache/raw_data/Weave_SDK_code/weave_sdk_code/documents.jsonl"
    ).readlines()
    source_document = json.loads(data_file[0])

    source_document = Document(**source_document)
    # print(source_document.page_content)

    code_transformer = CodeTextTransformer(
        lang_detect=lang_detect,
        chunk_size=1024,
    )

    transformed_documents = code_transformer.transform_documents(
        [source_document]
    )

    for document in transformed_documents:
        print(document.page_content)
        # print(document.metadata["source_content"])
        print(json.dumps(document.metadata, indent=2))
        print("*" * 80)
