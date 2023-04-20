import abc
import json
import logging
import pathlib
from typing import Optional, Union, List, Dict, Any

import joblib
import scipy.sparse
import tiktoken
from langchain.document_loaders import (
    UnstructuredMarkdownLoader,
    NotebookLoader,
    TextLoader,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import (
    TextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    TokenTextSplitter,
)
from llama_index import Document as LlamaDocument
from llama_index.docstore import DocumentStore as LlamaDocumentStore
from pydantic import BaseModel, BaseSettings, AnyHttpUrl, Field, root_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import wandb
from src.wandbot.customization.langchain import (
    ChromaWithEmbeddingsAndScores,
    HybridRetriever,
    TFIDFRetrieverWithScore,
)
from src.wandbot.ingestion.utils import (
    fetch_git_repo,
    md5_dir,
    add_metadata_to_documents,
)
from src.wandbot.prompts import load_hyde_prompt

logger = logging.getLogger(__name__)


class DataSource(BaseSettings):
    cache_dir: pathlib.Path = Field(
        pathlib.Path.home() / ".cache" / "wandbot", env="WANDBOT_CACHE_DIR"
    )
    ignore_cache: bool = False
    remote_path: Optional[Union[AnyHttpUrl, str]] = ""
    repo_path: Optional[Union[AnyHttpUrl, str]] = ""
    local_path: Optional[pathlib.Path] = None
    base_path: Optional[str] = ""
    file_pattern: str = "*.*"
    is_git_repo: bool = False
    git_id_file: Optional[pathlib.Path] = Field(None, env="WANDBOT_GIT_ID_FILE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DataStoreConfig(BaseModel):
    name: str = "docstore"
    data_source: DataSource = DataSource()
    chunk_size: int = 1024
    chunk_overlap: int = 0
    encoding_name: str = "cl100k_base"
    docstore_file: pathlib.Path = pathlib.Path("docstore.json")

    @root_validator(pre=False)
    def _set_cache_paths(cls, values: dict) -> dict:
        values["docstore_file"] = (
            values["data_source"].cache_dir / values["name"] / values["docstore_file"]
        )
        data_source = values["data_source"].dict()

        if data_source["repo_path"] and isinstance(
            data_source["repo_path"], AnyHttpUrl
        ):
            data_source["is_git_repo"] = data_source["repo_path"].host == "github.com"
            local_path = data_source["repo_path"].path.split("/")[-1]
            if not data_source["local_path"]:
                data_source["local_path"] = (
                    data_source["cache_dir"] / values["name"] / local_path
                )
            if data_source["is_git_repo"]:
                if data_source["git_id_file"] is None:
                    logger.warning(
                        "The source data is a git repo but no git_id_file is set."
                        " Attempting to use the default ssh id file"
                    )
                    data_source["git_id_file"] = pathlib.Path.home() / ".ssh" / "id_rsa"
        values["data_source"] = DataSource(**data_source)

        return values


class DocumentationStoreConfig(DataStoreConfig):
    name: str = "documentation_store"
    data_source: DataSource = DataSource(
        remote_path="https://docs.wandb.ai/",
        repo_path="https://github.com/wandb/docodile",
        base_path="",
        file_pattern="*.md",
        is_git_repo=True,
    )


class ExampleCodeStoreConfig(DataStoreConfig):
    name: str = "examples_code_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/examples/blob/master/",
        repo_path="https://github.com/wandb/examples",
        base_path="examples",
        file_pattern="*.py",
        is_git_repo=True,
    )


class ExampleNotebookStoreConfig(DataStoreConfig):
    name: str = "examples_notebook_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/examples/blob/master/",
        repo_path="https://github.com/wandb/examples",
        base_path="colabs",
        file_pattern="*.ipynb",
        is_git_repo=True,
    )


class SDKCodeStoreConfig(DataStoreConfig):
    name: str = "sdk_code_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/wandb/blob/main/",
        repo_path="https://github.com/wandb/wandb",
        base_path="wandb",
        file_pattern="*.py",
        is_git_repo=True,
    )


class ExtraDataStoreConfig(DataStoreConfig):
    name: str = "extra_data_store"
    data_source: DataSource = DataSource(
        local_path=pathlib.Path("data/raw_dataset/extra_data").resolve(),
        file_pattern="*.jsonl",
        is_git_repo=False,
    )


class DataStore:
    document_store: LlamaDocumentStore = None

    def __init__(self, config: DataStoreConfig):
        self.config = config
        self.md_text_splitter: TextSplitter = MarkdownTextSplitter()
        self.code_text_splitter: TextSplitter = PythonCodeTextSplitter()
        self.token_splitter: TextSplitter = TokenTextSplitter(
            encoding_name=self.config.encoding_name,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            allowed_special={"<|endoftext|>"},
        )

    def load_docstore_metadata(
        self,
    ):
        data_source = self.config.data_source
        if data_source.is_git_repo:
            doc_store_metadata = fetch_git_repo(data_source, data_source.git_id_file)
        else:
            doc_store_metadata = {
                "commit_hash": md5_dir(
                    data_source.local_path, file_pattern=data_source.file_pattern
                )
            }
        doc_store_metadata["source_name"] = self.config.name
        return doc_store_metadata

    def make_documents_tokenization_safe(self, documents):
        encoding = tiktoken.get_encoding(self.config.encoding_name)
        special_tokens_set = encoding.special_tokens_set

        def remove_special_tokens(text):
            for token in special_tokens_set:
                text = text.replace(token, "")
            return text

        cleaned_documents = []
        for document in documents:
            document = Document(
                page_content=remove_special_tokens(document.page_content),
                metadata=document.metadata,
            )
            cleaned_documents.append(document)
        return cleaned_documents

    def create_docstore_from_documents(
        self,
        documents: List[Document],
        source_map: Optional[Dict[str, str]] = None,
        metadata: Dict[str, str] = None,
    ) -> LlamaDocumentStore:
        documents = self.make_documents_tokenization_safe(documents)
        documents = self.token_splitter.split_documents(documents)
        documents = add_metadata_to_documents(documents, source_map)
        llama_documents: Dict[str, LlamaDocument] = {}
        for doc in documents:
            llama_documents[doc.metadata["doc_id"]] = LlamaDocument(
                text=doc.page_content,
                extra_info=doc.metadata,
                doc_id=doc.metadata["doc_id"],
            )
        if metadata is None:
            metadata = {}
        return LlamaDocumentStore(
            docs=llama_documents, ref_doc_info={"metadata": metadata}
        )

    @abc.abstractmethod
    def load_docstore(self) -> LlamaDocumentStore:
        raise NotImplementedError("Implement this in the subclass")

    def load(self) -> LlamaDocumentStore:
        docstore = self.load_docstore()
        return docstore


class DocumentationDataStore(DataStore):
    def load_docstore(
        self,
    ) -> LlamaDocumentStore:
        metadata = self.load_docstore_metadata()

        local_paths = (
            self.config.data_source.local_path / self.config.data_source.base_path
        ).rglob(self.config.data_source.file_pattern)
        dir_name = self.config.data_source.local_path.stem

        path_parts = map(lambda x: x.parts, local_paths)
        path_parts = list(filter(lambda x: len(x) > 2, path_parts))
        doc_indices = list(map(lambda x: x.index(dir_name), path_parts))

        local_paths = map(lambda x: str(pathlib.Path(*x)), path_parts)

        link_paths = map(
            lambda x: str(pathlib.Path(*x[1][(x[0] + 2) :]))[:-3],
            zip(doc_indices, path_parts),
        )
        link_paths = map(lambda x: x.replace("/other", ""), link_paths)
        link_paths = map(lambda x: x.replace("/intro", ""), link_paths)
        link_paths = map(lambda x: x.replace("/README", ""), link_paths)

        link_paths = map(
            lambda x: f"{self.config.data_source.remote_path}{x}", link_paths
        )

        document_files = dict(zip(local_paths, link_paths))

        documents = []
        for f_name in tqdm(
            document_files,
            desc=f"Loading documentation from {self.config.data_source.local_path}",
        ):
            try:
                documents.extend(UnstructuredMarkdownLoader(f_name).load())
            except:
                logger.warning(f"Failed to load documentation {f_name}")
        document_sections = self.md_text_splitter.split_documents(documents)

        document_store = self.create_docstore_from_documents(
            document_sections, document_files, metadata
        )
        return document_store


class CodeDataStore(DataStore):
    def load_docstore(self) -> LlamaDocumentStore:
        metadata = self.load_docstore_metadata()

        local_paths = (
            self.config.data_source.local_path / self.config.data_source.base_path
        ).rglob(self.config.data_source.file_pattern)

        paths = list(local_paths)
        local_paths = list(map(lambda x: str(x), paths))
        local_path_parts = list(map(lambda x: x.parts, paths))
        examples_idx = list(
            map(
                lambda x: x.index(self.config.data_source.local_path.stem),
                local_path_parts,
            )
        )
        remote_paths = list(
            map(
                lambda x: "/".join(x[1][x[0] + 1 :]),
                zip(examples_idx, local_path_parts),
            )
        )
        remote_paths = list(
            map(
                lambda x: f"{self.config.data_source.remote_path}{x}",
                remote_paths,
            )
        )
        document_files = dict(zip(local_paths, remote_paths))

        documents = []
        for f_name in tqdm(
            document_files,
            desc=f"Loading code from {self.config.data_source.local_path}",
        ):
            try:
                if self.config.data_source.file_pattern == "*.ipynb":
                    documents.extend(
                        NotebookLoader(
                            f_name,
                            include_outputs=False,
                            max_output_length=0,
                            remove_newline=True,
                        ).load()
                    )
                else:
                    documents.extend(TextLoader(f_name).load())
            except:
                logger.warning(f"Failed to load code in {f_name}")
        document_sections = self.code_text_splitter.split_documents(documents)
        document_store = self.create_docstore_from_documents(
            document_sections, document_files, metadata
        )
        return document_store


class ExtraDataStore(DataStore):
    def load_docstore(self) -> LlamaDocumentStore:
        metadata = self.load_docstore_metadata()

        jsonl_paths = (
            self.config.data_source.local_path / self.config.data_source.base_path
        ).rglob(self.config.data_source.file_pattern)

        all_documents = []
        for path in jsonl_paths:
            for line in path.open("r"):
                doc = json.loads(line)
                document = Document(
                    page_content=doc["document"], metadata={"source": doc["source"]}
                )
                all_documents.append(document)
        document_sections = self.md_text_splitter.split_documents(all_documents)
        document_store = self.create_docstore_from_documents(
            document_sections, None, metadata
        )
        return document_store


class VectorIndexConfig(BaseModel):
    name: str = "wandbot_vectorindex"
    cache_dir: pathlib.Path = Field(
        pathlib.Path.home() / ".cache" / "wandbot", env="WANDBOT_CACHE_DIR"
    )
    hyde_temperature: float = 0.3
    hyde_prompt: pathlib.Path = pathlib.Path("data/prompts/hyde_prompt.txt")

    vectorindex_dir: pathlib.Path = Field("vectorindex", env="WANDBOT_VECTORINDEX_DIR")
    sparse_vectorizer_kwargs: Dict[str, Any] = {
        "max_df": 0.9,
        "min_df": 0.1,
        "ngram_range": (1, 3),
    }
    wandb_project: str = "wandb_docs_bot"
    wandb_entity: Optional[str] = None

    class Config:
        env_prefix = "WANDBOT_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    @root_validator(pre=False)
    def _set_defaults(cls, values):
        if values["vectorindex_dir"] is not None:
            values["vectorindex_dir"] = (
                values["cache_dir"] / values["name"] / values["vectorindex_dir"]
            )
        return values


class VectorIndex:
    def __init__(self, config: VectorIndexConfig):
        self.config = config

        self.hyde_prompt = load_hyde_prompt(self.config.hyde_prompt)
        # self.embedding_fn = HypotheticalDocumentEmbedder(
        #     llm_chain=LLMChain(
        #         llm=ChatOpenAI(temperature=self.config.hyde_temperature),
        #         prompt=self.hyde_prompt,
        #     ),
        #     base_embeddings=OpenAIEmbeddings(),
        # )
        self.embedding_fn = OpenAIEmbeddings()

        self.datastore: LlamaDocumentStore = None
        self.retriever: HybridRetriever = None
        self.wandb_run = None

    def load_datastore(self, sources: List[DataStore]):
        datastore = {"metadata": {}, "docs": {}}
        for source in sources:
            data_dict = source.load()
            datastore["docs"] = dict(**datastore.get("docs", {}), **data_dict.docs)
            datastore["metadata"][
                data_dict.ref_doc_info["metadata"]["source_name"]
            ] = data_dict.ref_doc_info["metadata"]
        datastore = LlamaDocumentStore(
            docs=datastore["docs"], ref_doc_info={"metadata": datastore["metadata"]}
        )
        return datastore

    def create_retriever(self, datastore: LlamaDocumentStore):
        docs_list = []
        for doc_id, document in sorted(datastore.docs.items(), key=lambda x: x[0]):
            docs_list.append(document.to_langchain_format())
        sparse_vectorizer = TfidfVectorizer(**self.config.sparse_vectorizer_kwargs)
        sparse_vectors = sparse_vectorizer.fit_transform(
            [doc.page_content for doc in docs_list]
        )
        sparse_retriever = TFIDFRetrieverWithScore(
            vectorizer=sparse_vectorizer,
            docs=docs_list,
            tfidf_array=sparse_vectors,
            k=4,
        )

        dense_vectorstore = ChromaWithEmbeddingsAndScores(
            collection_name=self.config.name,
            persist_directory=str(self.config.vectorindex_dir / "dense_retriever"),
            embedding_function=self.embedding_fn,
            collection_metadata=datastore.ref_doc_info["metadata"],
        )
        dense_vectorstore.add_texts(
            texts=[doc.page_content for doc in docs_list],
            metadatas=[doc.metadata for doc in docs_list],
            ids=[doc.metadata["doc_id"] for doc in docs_list],
        )
        return HybridRetriever(
            dense=dense_vectorstore.as_retriever(), sparse=sparse_retriever
        )

    def load(self, data_sources: List[DataStore]) -> "VectorIndex":
        self.datastore = self.load_datastore(data_sources)
        self.retriever = self.create_retriever(self.datastore)
        return self

    def save(self):
        self.config.vectorindex_dir.mkdir(parents=True, exist_ok=True)
        # dump the datastore
        datastore_dict = self.datastore.serialize_to_dict()
        with open(self.config.vectorindex_dir / "datastore.json", "w") as f:
            json.dump(datastore_dict, f)
        with open(self.config.vectorindex_dir / "metadata.json", "w") as f:
            json.dump(self.datastore.ref_doc_info["metadata"], f)

        # dump the sparse retriever
        sparse_retriever_dir = self.config.vectorindex_dir / "sparse_retriever"
        sparse_retriever_dir.mkdir(parents=True, exist_ok=True)
        with open(sparse_retriever_dir / "sparse_vectorizer.pkl", "wb") as f:
            joblib.dump(self.retriever.sparse.vectorizer, f)
        scipy.sparse.save_npz(
            str(sparse_retriever_dir / "tfidf_array.npz"),
            self.retriever.sparse.tfidf_array,
        )

        # dump the hyde prompt
        with open(self.config.vectorindex_dir / "hyde_prompt.txt", "w") as f:
            f.write(self.config.hyde_prompt.open("r").read())

        # dump the dense retriever
        self.retriever.dense.vectorstore.persist()

        if self.wandb_run is None:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.dict(),
            )
        # dump the config
        with open(self.config.vectorindex_dir / "config.json", "w") as f:
            f.write(self.config.json())

        artifact = wandb.Artifact(
            name=self.config.name, type="vectorindex", metadata=self.config.dict()
        )

        artifact.add_dir(str(self.config.vectorindex_dir))
        self.wandb_run.log_artifact(artifact)
        return self

    def load_from_artifact(
        self, artifact_path: Optional[str] = None, version: str = "latest"
    ):
        if self.wandb_run is None:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.dict(),
            )
        if artifact_path is None:
            artifact_path = f"{self.wandb_run.entity}/{self.wandb_run.project}/{self.config.name}:{version}"
        artifact = self.wandb_run.use_artifact(artifact_path)
        artifact_dir = pathlib.Path(artifact.download())

        # load the config
        with open(artifact_dir / "config.json", "r") as f:
            config_dict = json.load(f)
        self.config = VectorIndexConfig(**config_dict)

        # load the hyde prompt
        self.hyde_prompt = load_hyde_prompt(artifact_dir / "hyde_prompt.txt")

        # TODO: load the embedding_fn from the hyde prompt

        # load the datastore
        with open(artifact_dir / "datastore.json", "r") as f:
            datastore_dict = json.load(f)
        self.datastore = LlamaDocumentStore.load_from_dict(datastore_dict)

        # load the metadata
        with open(artifact_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # load the sparse retriever
        sparse_retriever_dir = artifact_dir / "sparse_retriever"
        with open(sparse_retriever_dir / "sparse_vectorizer.pkl", "rb") as f:
            sparse_vectorizer = joblib.load(f)
        tfidf_array = scipy.sparse.load_npz(
            str(sparse_retriever_dir / "tfidf_array.npz"),
        )
        docs_list = []
        for doc_id, document in sorted(self.datastore.docs.items(), key=lambda x: x[0]):
            docs_list.append(document.to_langchain_format())
        sparse_retriever = TFIDFRetrieverWithScore(
            vectorizer=sparse_vectorizer,
            docs=docs_list,
            tfidf_array=tfidf_array,
            k=4,
        )

        # load the dense retriever
        dense_retriever_dir = str(artifact_dir / "dense_retriever")
        dense_vectorstore = ChromaWithEmbeddingsAndScores(
            persist_directory=dense_retriever_dir,
            embedding_function=self.embedding_fn,
            collection_name=self.config.name,
            collection_metadata=metadata,
        )
        self.retriever = HybridRetriever(
            sparse=sparse_retriever, dense=dense_vectorstore.as_retriever()
        )
        return self


def main():
    data_sources = [
        DocumentationDataStore(DocumentationStoreConfig()),
        CodeDataStore(ExampleCodeStoreConfig()),
        CodeDataStore(ExampleNotebookStoreConfig()),
        CodeDataStore(SDKCodeStoreConfig()),
        ExtraDataStore(ExtraDataStoreConfig()),
    ]
    # print(data_sources[1].load_docstore())
    vectorindex_config = VectorIndexConfig(wandb_project="wandb_docs_bot_dev")
    print(vectorindex_config.json(indent=2))
    vector_index = VectorIndex(vectorindex_config)
    # vector_index = vector_index.load(data_sources)
    # vector_index = vector_index.save()
    vector_index = vector_index.load_from_artifact()
    # vector_index = vector_index.save()
    print(vector_index.retriever.sparse.tfidf_array)


if __name__ == "__main__":
    main()
