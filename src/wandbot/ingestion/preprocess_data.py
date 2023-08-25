import hashlib
import json
import os
import pathlib
from typing import List

import tiktoken
import wandb
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from pydantic import BaseModel
from wandbot.utils import get_logger

logger = get_logger(__name__)


class DocumentTransformerConfig(BaseModel):
    cache_dir: pathlib.Path = pathlib.Path("data/cache/transformed_data")
    chunk_size: int = 2048
    chunk_overlap: int = 128
    encoding_type: str = "cl100k_base"
    add_start_index: bool = True


class DocumentTransformer:
    def __init__(self, config: DocumentTransformerConfig):
        self.config = config
        self.splitter_kwargs = dict(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=self.config.add_start_index,
        )
        self.document_splitters = {
            ".md": RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN, **self.splitter_kwargs
            ),
            ".py": RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON, **self.splitter_kwargs
            ),
            ".ipynb": RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON, **self.splitter_kwargs
            ),
            ".js": RecursiveCharacterTextSplitter.from_language(
                Language.JS, **self.splitter_kwargs
            ),
            ".ts": RecursiveCharacterTextSplitter.from_language(
                Language.JS, **self.splitter_kwargs
            ),
        }

    def make_documents_tokenization_safe(self, documents):
        encoding = tiktoken.get_encoding(self.config.encoding_type)
        special_tokens_set = encoding.special_tokens_set

        def remove_special_tokens(text):
            for token in special_tokens_set:
                text = text.replace(token, "")
            return text

        cleaned_documents = []
        for document in documents:
            document.page_content = remove_special_tokens(document.page_content)
            cleaned_documents.append(document)
        return cleaned_documents

    def transform(self, documents: List[Document]) -> List[Document]:
        transformed_documents = []
        documents = self.make_documents_tokenization_safe(documents)
        for document in documents:
            document_extension = document.metadata["file_type"]
            document_splitter = self.document_splitters.get(document_extension, None)
            if document_splitter is None:
                continue
            for doc_split in document_splitter.split_documents([document]):
                doc_split.metadata["parent_hash"] = document.metadata["hash"]
                doc_split.metadata["hash"] = doc_split.metadata["hash"] = hashlib.md5(
                    (
                        str(doc_split.metadata["parent_hash"])
                        + str(doc_split.page_content)
                    ).encode("UTF-8")
                ).hexdigest()
                transformed_documents.append(doc_split)
        return transformed_documents


def load(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = "transformed_dataset",
):
    run = wandb.init(project=project, entity=entity, job_type="preprocess_dataset")
    artifact = run.use_artifact(source_artifact_path, type="dataset")
    artifact_dir = artifact.download()
    processed_artifact = wandb.Artifact(
        result_artifact_name,
        type="dataset",
        description="Wandbot dataset with transformed documents",
    )
    document_files = list(pathlib.Path(artifact_dir).rglob("documents.jsonl"))
    transformer = DocumentTransformer(DocumentTransformerConfig())
    for document_file in document_files:
        documents = []
        with document_file.open() as f:
            for line in f:
                doc_dict = json.loads(line)
                doc = Document(**doc_dict)
                documents.append(doc)
        transformed_documents = transformer.transform(documents)
        doc_store_dir = transformer.config.cache_dir.joinpath(document_file.parent.name)
        doc_store_dir.mkdir(parents=True, exist_ok=True)
        with doc_store_dir.joinpath("documents.jsonl").open("w") as f:
            for document in transformed_documents:
                document = {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                f.write(json.dumps(document) + "\n")
        doc_store_dir.joinpath("config.json").write_text(
            transformer.config.model_dump_json()
        )
        metadata = {
            "num_documents": len(documents),
            "num_transformed_documents": len(transformed_documents),
        }
        doc_store_dir.joinpath("metadata.json").write_text(json.dumps(metadata))
        processed_artifact.add_dir(str(doc_store_dir), name=doc_store_dir.name)
    run.log_artifact(processed_artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"


def main():
    load(
        project=os.environ.get("WANDB_PROJECT", "wandbot-dev"),
        entity=os.environ.get("WANDB_ENTITY", "wandbot"),
        source_artifact_path="wandbot/wandbot-dev/raw_dataset:latest",
    )


if __name__ == "__main__":
    main()
