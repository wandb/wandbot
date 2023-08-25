import json
import pathlib
from typing import List

import wandb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import TFIDFRetriever
from langchain.schema import Document
from langchain.vectorstores import FAISS
from pydantic import BaseModel
from wandbot.utils import get_logger

logger = get_logger(__name__)


class RetrieverConfig(BaseModel):
    cache_dir: pathlib.Path = pathlib.Path("data/cache/retreiver_data")
    tfidf_params: dict = {
        "max_df": 0.9,
        "ngram_range": (1, 3),
    }


class RetrieverLoader:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.embedding_fn = OpenAIEmbeddings()

    def create_faiss_store(
        self, docstore: pathlib.Path, documents: List[Document]
    ) -> pathlib.Path:
        persist_directory = docstore / "faiss"
        persist_directory.mkdir(parents=True, exist_ok=True)
        faiss = FAISS.from_documents(
            documents,
            self.embedding_fn,
        )
        faiss.save_local(str(persist_directory))
        return persist_directory

    def create_tfidf_store(
        self, docstore: pathlib.Path, documents: List[Document]
    ) -> pathlib.Path:
        persist_directory = docstore / "tfidf"
        persist_directory.mkdir(parents=True, exist_ok=True)
        tfidf = TFIDFRetriever.from_documents(
            documents, tfidf_params=self.config.tfidf_params
        )
        tfidf.save_local(str(persist_directory))
        return persist_directory

    def create_retrievers(
        self, docstore: str, documents: List[Document]
    ) -> pathlib.Path:
        persist_directory = self.config.cache_dir / docstore
        _ = self.create_faiss_store(persist_directory, documents)
        _ = self.create_tfidf_store(persist_directory, documents)
        persist_directory.joinpath("config.json").write_text(self.config.json())
        metadata = {"num_documents": len(documents)}
        persist_directory.joinpath("metadata.json").write_text(json.dumps(metadata))
        return persist_directory


def load(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = "vectorstores",
):
    run = wandb.init(project=project, entity=entity, job_type="vectorized_dataset")
    artifact = run.use_artifact(source_artifact_path, type="dataset")
    artifact_dir = artifact.download()
    retriever_artifact = wandb.Artifact(
        result_artifact_name,
        type="vectorstore",
        description="Wandbot VectorStore Retrievers with transformed documents",
    )
    retriever_loader = RetrieverLoader(RetrieverConfig())
    document_files = list(pathlib.Path(artifact_dir).rglob("documents.jsonl"))
    for document_file in document_files:
        documents = []
        with document_file.open() as f:
            for line in f:
                doc_dict = json.loads(line)
                doc = Document(**doc_dict)
                documents.append(doc)

        retrievers_dir = retriever_loader.create_retrievers(
            document_file.parent.name, documents
        )
        retriever_artifact.add_dir(str(retrievers_dir), name=retrievers_dir.name)
    run.log_artifact(retriever_artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"


def main():
    load(
        project="wandbot-dev",
        entity="wandbot",
        source_artifact_path="wandbot/wandbot-dev/transformed_dataset:latest",
    )


if __name__ == "__main__":
    main()
