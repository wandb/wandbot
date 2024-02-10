import json
from typing import Any, Callable, List, Sequence

from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import BaseDocumentTransformer, Document
from tree_sitter import Node
from tree_sitter_languages import get_language, get_parser
from wandbot.ingestion.preprocess.markdown import (
    CustomMarkdownTextSplitter,
    create_id_from_document,
)
from wandbot.utils import FastTextLangDetect, FasttextModelConfig


def extract_docstrings(node: Node, node_type: str, language: Any):
    """
    Extracts docstrings for modules, classes, and functions using Tree-sitter queries.
    Also captures parent node details for naming.

    Args:
    - node: The root node of the syntax tree.
    - source_bytes: The source code as bytes.

    Returns:
    - A list of tuples, each with the docstring node, its parent node, and the capture name.
    """
    module_doc_str_pattern = "(module . (comment)* . (expression_statement (string)) @module_doc_str)"

    class_doc_str_pattern = "(class_definition body: (block . (expression_statement (string)) @class_doc_str))"
    function_doc_str_pattern = "(function_definition body: (block . (expression_statement (string)) @function_doc_str))"
    if node_type in ["decorated_function", "decorated_method", "function"]:
        doc_str_pattern = function_doc_str_pattern
    elif node_type in ["decorated_class", "class"]:
        doc_str_pattern = class_doc_str_pattern
    else:
        doc_str_pattern = module_doc_str_pattern

    doc_str_query = language.query(doc_str_pattern)
    doc_strs = doc_str_query.captures(node)
    return doc_strs


def traverse(
    node: Node,
    source_bytes: bytes,
    context_stack: List[str] | None = None,
    max_length: int = None,
    language: Any = None,
):
    if context_stack is None:
        context_stack = []

    definitions = []
    for child in node.children:
        if child.type in [
            "class_definition",
            "function_definition",
            "decorated_definition",
        ]:
            name_node = next(
                filter(lambda n: n.type == "identifier", child.children),
                None,
            )
            name = name_node.text.decode() if name_node else "Unnamed"
            name_delimiter = "!"
            is_method = any(context == "class" for context in context_stack)
            type_name = "Unknown"
            is_decorated = False

            body_start = child.start_byte
            body_end = child.end_byte
            body_length = len(source_bytes[body_start:body_end])

            if child.type == "decorated_definition":
                # capture whether it is a class, method or function
                is_class = next(
                    filter(
                        lambda n: n.type == b"class_definition", child.children
                    ),
                    None,
                )
                if is_class:
                    type_name = "decorated_class"
                else:
                    type_name = (
                        "decorated_method"
                        if is_method
                        else "decorated_function"
                    )
                decorator_node = next(
                    filter(lambda n: n.type == "decorator", child.children),
                    None,
                )
                decorator_name = (
                    decorator_node.text.decode()
                    if decorator_node
                    else "Unnamed"
                )
                decorated_node = next(
                    filter(
                        lambda n: n.type
                        in ["function_definition", "class_definition"],
                        child.children,
                    ),
                    None,
                )
                decorated_name_node = next(
                    filter(
                        lambda n: n.type == "identifier",
                        decorated_node.children,
                    ),
                    None,
                )

                decorated_name = (
                    decorated_name_node.text.decode()
                    if decorated_node
                    else "Unnamed"
                )

                if decorated_node.type == "class_definition":
                    decorated_type_name = "class"
                elif decorated_node.type == "function_definition":
                    decorated_type_name = "method" if is_method else "function"
                else:
                    decorated_type_name = "Unknown"

                name = f"{decorator_name}{name_delimiter}{type_name}{name_delimiter}{decorated_name}"
                type_name = decorated_type_name

                full_name = name_delimiter.join(
                    context_stack + [name, type_name]
                )

                body_start = child.start_byte
                body_end = child.end_byte
                body_length = len(source_bytes[body_start:body_end])
                child = decorated_node
                is_decorated = True
            if child.type == "class_definition":
                type_name = "class"
                full_name = name_delimiter.join(
                    context_stack + [name, type_name]
                )

            elif child.type == "function_definition":
                type_name = "method" if is_method else "function"
                full_name = name_delimiter.join(
                    context_stack + [name, type_name]
                )
            else:
                continue

            if is_decorated:
                type_name = "decorated_" + type_name
            # Check if the body length exceeds chunk_size before extracting docstrings
            if max_length is None or body_length > max_length:
                # Extract and handle docstrings
                doc_strs = extract_docstrings(child, type_name, language)
                for doc_str, capture_name in doc_strs:
                    docstring_start = doc_str.start_byte
                    docstring_end = doc_str.end_byte
                    definitions.append(
                        {
                            "type": "docstring",
                            "name": f"{full_name}{name_delimiter}{capture_name}{name_delimiter}docstring",
                            "body": (docstring_start, docstring_end),
                        }
                    )

                definitions.append(
                    {
                        "type": type_name,
                        "name": f"{full_name}",
                        "body": (body_start, body_end),
                    }
                )

                new_context_stack = context_stack.copy()
                new_context_stack.append(name)
                new_context_stack.append(type_name)

                # Recursively traverse child nodes with updated context
                definitions.extend(
                    traverse(
                        child,
                        source_bytes,
                        context_stack=new_context_stack,
                        max_length=max_length,
                        language=language,
                    )
                )
            else:
                # If the body is smaller than chunk_size, treat it as a leaf node without extracting docstrings
                definitions.append(
                    {
                        "type": type_name,
                        "name": full_name,
                        "body": (body_start, body_end),
                    }
                )
        else:
            # Recursively traverse child nodes without changing the context
            definitions.extend(
                traverse(
                    child,
                    source_bytes,
                    context_stack=context_stack,
                    max_length=max_length,
                    language=language,
                )
            )
    return definitions


def get_line_number(index: int, source_code: bytes) -> int:
    total_chars = 0
    for line_number, line in enumerate(
        source_code.splitlines(keepends=True), start=1
    ):
        total_chars += len(line)
        if total_chars > index:
            return line_number - 1
    return line_number


def load_definitions(definitions, sourcecode_bytes):
    line_definitions = []
    for definition in definitions:
        start, end = definition["body"]
        start_line = get_line_number(start, sourcecode_bytes)
        end_line = get_line_number(end, sourcecode_bytes)
        definition["span"] = [start_line, end_line]
        definition["body"] = b"\n".join(
            sourcecode_bytes.splitlines()[start_line : end_line + 1]
        ).decode("utf-8")

        line_definitions.append(definition)
    return line_definitions


def post_process_definitions(definitions, name_delimiter="!"):
    """
    Post-processes the list of definitions to remove the bodies of child nodes
    from their parent nodes and replace them with placeholders, maintaining the indent level.

    Args:
    - definitions: The list of dictionaries representing class or function definitions.

    Returns:
    - The modified list of definitions with child bodies removed from parent bodies and placeholders indented appropriately.
    """
    # Sort definitions by name to ensure parents are processed before their children
    definitions.sort(key=lambda x: x["name"])

    for i, defn in enumerate(definitions):
        full_name = defn["name"]
        body = defn["body"]

        # Look for children of the current definition
        for child in definitions[i + 1 :]:
            child_full_name = child["name"]
            # Check if the current definition is a direct parent of the child
            if child_full_name.startswith(full_name + name_delimiter):
                child_body = child["body"]
                # Calculate the indent by looking backwards from the start_index to find the newline character
                indent = " " * (len(child_body) - len(child_body.lstrip()))

                placeholder_child_name = ":".join(
                    name
                    for name in child_full_name.split(name_delimiter)
                    if name
                    not in [
                        "class",
                        "function",
                        "method",
                        "docstring",
                        "decorated_class",
                        "decorated_function",
                        "decorated_method",
                    ]
                )
                placeholder = f"{indent}#{child['name']}:"
                # placeholder = ""
                # Replace the child's body in the parent's body with an indented placeholder
                body = body.replace(
                    child_body, placeholder, 1
                )  # Replace only the first occurrence

        # Update the parent body after all replacements
        defn["body"] = body

    return definitions


def has_more_than_n_imports(root_node, n):
    """
    Checks if the Python module represented by the Tree-sitter root node
    contains more than n import statements, returning early if so.

    Args:
    - root_node: The root node of the Tree-sitter syntax tree for the module.
    - n: The threshold number of import statements.

    Returns:
    - A boolean indicating whether the module has more than n import statements.
    """
    import_count = 0

    # Function to recursively traverse the tree and count import statements
    def count_imports(node):
        nonlocal import_count
        if import_count > n:
            # Return early if we've already found more than n imports
            return True
        for child in node.children:
            if child.type in ["import_statement", "import_from_statement"]:
                import_count += 1
                if import_count > n:
                    # Found more than n imports, can return early
                    return True
            # Continue the traversal until more than n imports are found
            if count_imports(child):
                return True
        return False

    # Start the traversal from the root node
    return count_imports(root_node)


def check_merge(definitions, max_length):
    """Checks if a list of definitions can be merged, considering docstrings and chunk_size."""
    # Check for docstring type
    if any(defn["type"] == "docstring" for defn in definitions):
        return False
    # Check total length
    total_length = sum(len(defn["body"]) for defn in definitions)
    return total_length <= max_length


def get_parent(definition):
    """Returns the parent path from a definition's name, considering the definition's type for decorated nodes."""
    parts = definition["name"].split("!")
    # Determine if the definition is for a decorated node based on its type
    if any(
        part.startswith("decorated_")
        for part in definition["type"]
        if isinstance(definition["type"], list)
    ) or definition["type"].startswith("decorated_"):
        return "!".join(parts[:-4])  # Adjust for decorated nodes
    else:
        return "!".join(parts[:-2])  # Standard parent extraction


# Correct the merge_definitions function to handle the merging correctly
def merge_definitions(definitions, max_length):
    merged_definitions = []
    definitions_by_parent = {}

    # Group definitions by their parent path, considering decorated nodes
    for definition in definitions:
        parent = get_parent(definition)
        if parent not in definitions_by_parent:
            definitions_by_parent[parent] = []
        definitions_by_parent[parent].append(definition)

    # Attempt to merge definitions within the same parent group
    for parent, defs in definitions_by_parent.items():
        while defs:
            current = defs.pop(0)
            if (
                "docstring" in current["type"]
                or len(current["body"]) > max_length
            ):
                # Handle docstrings and oversized definitions individually, ensuring they are not merged
                current["type"], current["name"], current["span"] = (
                    [current["type"]],
                    [current["name"]],
                    [current["span"]],
                )
                merged_definitions.append(current)
                continue
            to_merge = [current]
            for defn in list(defs):
                if (
                    "docstring" in defn["type"]
                    or len(defn["body"]) > max_length
                ):
                    continue  # Skip non-eligible definitions
                if check_merge(to_merge + [defn], max_length):
                    to_merge.append(defn)
                    defs.remove(defn)
            if len(to_merge) > 1:
                # Merge the definitions
                merged_definition = {
                    "type": [defn["type"] for defn in to_merge],
                    "name": [defn["name"] for defn in to_merge],
                    "span": [defn["span"] for defn in to_merge],
                    "body": "\n\n".join(defn["body"] for defn in to_merge),
                }
                merged_definitions.append(merged_definition)
            else:
                # No merge occurred, adjust keys to lists for the single definition
                current["type"], current["name"], current["span"] = (
                    [current["type"]],
                    [current["name"]],
                    [current["span"]],
                )
                merged_definitions.append(current)

    return merged_definitions


def chunk_source(
    root_node: Node,
    source_code: bytes,
    max_length: int = 512,
    language: Any = None,
):
    definitions = traverse(
        root_node, source_code, None, max_length=max_length, language=language
    )
    definitions = load_definitions(definitions, source_code)
    definitions = post_process_definitions(definitions)

    definitions = merge_definitions(definitions, max_length)

    return definitions


def get_text_from_definition(definition):
    final_definition = f"""
source_types:\t{definition["type"]}
source_paths:\t{definition["name"]}
source_lines:\t{definition["span"]}
{definition["body"]}
"""
    return final_definition


class PythonCodeTextTransformer(BaseDocumentTransformer):
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
        self.parser = get_parser("python")
        self.language = get_language("python")

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        transformed_documents = []
        for document in list(documents):
            source_code = document.page_content.encode("utf-8")
            tree = self.parser.parse(source_code)
            definitions = chunk_source(
                tree.root_node,
                source_code,
                max_length=self.chunk_size,
                language=self.language,
            )
            for definition in definitions:
                transformed_documents.append(
                    Document(
                        page_content=get_text_from_definition(definition),
                        metadata=document.metadata,
                    )
                )
        return transformed_documents


class CodeTextTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        lang_detect,
        chunk_size: int = 512,
        length_function: Callable[[str], int] = None,
    ):
        self.lang_detect = lang_detect
        self.chunk_size: int = chunk_size
        self.length_function: Callable[[str], int] = (
            length_function if length_function is not None else len
        )
        self.lang_detect = None
        self.python_code_splitter = PythonCodeTextTransformer(
            chunk_size=self.chunk_size * 2
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split incoming code and return chunks using the AST."""

        # First chunk into parent documents

        chunked_documents = []
        for document in documents:
            file_extension = document.metadata.get("file_type", "")
            if file_extension in [".py", ".js", ".ts"]:
                language = {
                    ".py": Language.PYTHON,
                    ".js": Language.JS,
                    ".ts": Language.JS,
                }[file_extension]
                if language == Language.PYTHON:
                    chunked_documents.extend(
                        self.python_code_splitter.transform_documents(
                            [document]
                        )
                    )
                else:
                    recursive_splitter = (
                        RecursiveCharacterTextSplitter.from_language(
                            language=language,
                            chunk_size=self.chunk_size * 2,
                            chunk_overlap=0,
                            keep_separator=True,
                            length_function=len,
                        )
                    )
                    chunked_documents.extend(
                        recursive_splitter.split_documents([document])
                    )
            elif file_extension in [".md", ".ipynb"]:
                chunked_documents.extend(
                    CustomMarkdownTextSplitter(
                        chunk_size=self.chunk_size * 2
                    ).split_documents([document])
                )
            else:
                chunked_documents.extend(
                    TokenTextSplitter(
                        chunk_size=self.chunk_size * 2
                    ).split_documents([document])
                )

        # make new documents from the chunks with updated metadata
        parent_documents = []
        for split in chunked_documents:
            file_extension = split.metadata.get("file_type", "")
            if file_extension == ".py":
                document_content = "\n".join(
                    split.page_content.strip().split("\n")[3:]
                )
            else:
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
            if file_extension in [".py", ".js", ".ts", ".md", ".ipynb"]:
                language = {
                    ".py": Language.PYTHON,
                    ".js": Language.JS,
                    ".ts": Language.JS,
                    ".md": Language.MARKDOWN,
                    ".ipynb": Language.MARKDOWN,
                }[file_extension]
                recursive_splitter = (
                    RecursiveCharacterTextSplitter.from_language(
                        language=language,
                        chunk_size=self.chunk_size,
                        chunk_overlap=0,
                        keep_separator=True,
                        length_function=self.length_function,
                    )
                )
                final_chunks.extend(
                    recursive_splitter.split_documents([document])
                )
            else:
                final_chunks.extend(
                    TokenTextSplitter(
                        chunk_size=self.chunk_size
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
        "/media/mugan/data/wandb/projects/wandbot/data/cache/raw_data/docodile_store/docodile_en/documents.jsonl"
    ).readlines()
    source_document = json.loads(data_file[0])

    source_document = Document(**source_document)
    print(source_document.page_content)

    code_transformer = CodeTextTransformer(
        lang_detect=lang_detect,
        chunk_size=768 // 2,
    )

    transformed_documents = code_transformer.transform_documents(
        [source_document]
    )

    for document in transformed_documents:
        print(document.page_content)
        # print(document.metadata["source_content"])
        print(json.dumps(document.metadata, indent=2))
        print("*" * 80)
