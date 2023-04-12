import abc
import hashlib
import json
import logging
import pathlib
import sys
from typing import List, Dict, Optional, Any, Union

import pandas as pd
import tiktoken
import wandb
from langchain import LLMChain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chat_models import ChatOpenAI
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
from langchain.vectorstores import Chroma
from llama_index import Document as LlamaDocument
from llama_index.docstore import DocumentStore as LlamaDocumentStore
from llama_index.schema import BaseDocument
from tqdm import tqdm

from src.wandbot.ingestion.settings import (
    DocStoreConfig,
    BaseDataConfig,
    CombinedDocStoreConfig,
    DocumentationStoreConfig,
    CodeStoreConfig,
    ExamplesCodeStoreConfig,
    ExamplesNotebookStoreConfig,
    SDKCodeStoreConfig,
    CsvStoreConfig,
    JsonlStoreConfig,
    ExtraDataStoreConfig,
    WandbotDocStoreConfig,
)
from src.wandbot.ingestion.utils import (
    md5_dir,
    ChromaWithEmbeddings,
    fetch_git_repo,
    map_local_to_remote,
    add_metadata_to_documents,
    convert_llama_docstore_to_vectorstore_kwargs,
    load_docstore_class,
)
from src.wandbot.prompts import load_hyde_prompt

logger = logging.getLogger(__name__)


class BaseDocStore:
    config: DocStoreConfig = DocStoreConfig()

    def __init__(self, config: Optional[DocStoreConfig] = None, **kwargs):
        if config is not None:
            self.config = config

        self.document_store = None
        self.vectorstore = None
        self.wandb_run = None
        self.hyde_prompt = load_hyde_prompt(self.config.hyde_prompt)
        self.embedding_fn = HypotheticalDocumentEmbedder(
            llm_chain=LLMChain(
                llm=ChatOpenAI(temperature=self.config.hyde_temperature),
                prompt=self.hyde_prompt,
            ),
            base_embeddings=OpenAIEmbeddings(),
        )

    def maybe_create_run(
        self,
    ):
        if self.wandb_run is None:
            self.wandb_run = (
                wandb.init(
                    project=self.config.wandb_project, entity=self.config.wandb_entity
                )
                if wandb.run is None
                else wandb.run
            )
        return self.wandb_run

    def maybe_save_docstore(
        self,
        document_store: LlamaDocumentStore,
        docstore_file: pathlib.Path = None,
        ignore_cache: bool = None,
    ):
        if ignore_cache is None:
            ignore_cache = self.config.data_config.ignore_cache
        if docstore_file is None:
            docstore_file = self.config.docstore_file

        doc_store_dict = document_store.serialize_to_dict()
        if docstore_file.is_file() and ignore_cache:
            logger.debug(f"{docstore_file} was found, but ignoring cache")
            json.dump(
                doc_store_dict,
                open(docstore_file, "w"),
            )
        if not docstore_file.is_file():
            logger.debug(f"{docstore_file} was not found, writing to cache")
            docstore_file.parent.mkdir(parents=True, exist_ok=True)
            json.dump(
                doc_store_dict,
                open(docstore_file, "w"),
            )
        return document_store

    def maybe_save_vectorstore(
        self,
        vectorstore: Chroma,
        vectorstore_dir: pathlib.Path = None,
        ignore_cache=None,
    ):
        if ignore_cache is None:
            ignore_cache = self.config.data_config.ignore_cache
        if vectorstore_dir is None:
            vectorstore_dir = pathlib.Path(self.config.vectorstore_dir)
        if vectorstore_dir.exists() and ignore_cache:
            logger.debug(f"{vectorstore_dir} was found, but overwriting cache")
            vectorstore.persist()
        elif not vectorstore_dir.exists():
            logger.debug(f"{vectorstore_dir} was not found, writing to cache")
            vectorstore.persist()
        return vectorstore

    @abc.abstractmethod
    def load_docstore(*args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def load_vectorstore(*args, **kwargs):
        raise NotImplementedError

    def save(self):
        self.wandb_run = self.maybe_create_run()
        self.vectorstore = self.load_vectorstore(self.config.data_config)
        artifact_metadata = json.loads(self.config.json())
        artifact_metadata["doc_store_info"] = self.document_store.ref_doc_info
        artifact_metadata["num_docs"] = len(self.document_store.docs)

        artifact = wandb.Artifact(
            self.config.name, type=self.config.type, metadata=artifact_metadata
        )
        artifact.add_file(
            str(self.config.docstore_file), name=self.config.docstore_file.name
        )
        artifact.add_dir(
            str(self.config.vectorstore_dir), name=self.config.vectorstore_dir.name
        )
        with artifact.new_file(self.config.hyde_prompt.name, "w+") as f:
            f.write(self.hyde_prompt.messages[0].prompt.template)
        self.wandb_run.log_artifact(artifact)
        return self

    def load_from_artifact(self, artifact_path: str):
        artifact = self.wandb_run.use_artifact(artifact_path)
        artifact_dir = artifact.download()
        artifact_dir = pathlib.Path(artifact_dir)

        self.config.docstore_file = artifact_dir / self.config.docstore_file.name
        self.config.vectorstore_dir = artifact_dir / self.config.vectorstore_dir.name
        self.hyde_prompt = load_hyde_prompt(
            str(artifact_dir / self.config.hyde_prompt.name)
        )
        self.document_store = self.load_docstore(self.config.data_config)
        self.vectorstore = self.load_vectorstore()
        return self

    def verify_and_load_artifact(
        self, artifact_path, artifact_commit_hash, source_commit_hash
    ):
        if artifact_commit_hash is not None:
            if artifact_commit_hash != source_commit_hash:
                logger.warning(
                    f"Artifact commit hash {artifact_commit_hash} does not match source commit hash "
                    f"{source_commit_hash}, recreating artifact"
                )
                return self.save()
            else:
                logger.debug(
                    f"Artifact commit hash {artifact_commit_hash} matches source commit hash "
                    f"{source_commit_hash}, loading artifact"
                )
                return self.load_from_artifact(artifact_path)
        else:
            logger.debug(f"Artifact commit hash not found, recreating artifact")
            return self.save()

    @abc.abstractmethod
    def maybe_load_from_artifact(self, artifact_path: str = None, version="latest"):
        raise NotImplementedError

    def load(self, artifact_path: str = None, version="latest"):
        return self.maybe_load_from_artifact(artifact_path, version=version)


class DocStore(BaseDocStore):
    config = DocStoreConfig()

    def __init__(self, config: Optional[DocStoreConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        if config is not None:
            self.config = config
        self.md_text_splitter: TextSplitter = MarkdownTextSplitter()
        self.code_text_splitter: TextSplitter = PythonCodeTextSplitter()
        self.token_splitter: TextSplitter = TokenTextSplitter(
            encoding_name=self.config.encoding_name,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            allowed_special={"<|endoftext|>"},
        )

    @abc.abstractmethod
    def _load_documents(
        self, paths: BaseDataConfig, docstore_metadata: Dict[str, str] = None
    ):
        raise NotImplementedError(
            "You need to implement this method in the subclass to parse and load your documents"
        )

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

    def load_docstore_metadata(self, paths: BaseDataConfig):
        if paths.is_git_repo:
            doc_store_metadata = fetch_git_repo(
                paths, self.config.data_config.git_id_file
            )
        else:
            doc_store_metadata = {
                "commit_hash": md5_dir(
                    paths.local_path, file_pattern=paths.file_pattern
                )
            }
        doc_store_metadata["source_name"] = self.config.name
        doc_store_metadata["source_cls"] = self.config.cls
        return doc_store_metadata

    def create_docstore(
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

    def maybe_load_docstore(self, paths=None):
        docstore_file = pathlib.Path(self.config.docstore_file)
        ignore_cache = self.config.data_config.ignore_cache
        source_metadata = self.load_docstore_metadata(paths)
        if docstore_file.is_file() and not ignore_cache:
            logger.debug(f"{docstore_file} was found, loading existing docs")
            doc_store_dict = json.load(open(docstore_file))
            cache_document_store = LlamaDocumentStore.load_from_dict(doc_store_dict)
            cache_commit_hash = cache_document_store.ref_doc_info.get(
                "metadata", {}
            ).get("commit_hash")
            if cache_commit_hash is not None:
                source_commit_hash = source_metadata.get("commit_hash")
                if cache_commit_hash != source_commit_hash:
                    ignore_cache = ignore_cache or True
                    logger.debug(
                        f"cache_commit_hash: {cache_commit_hash}, source_commit_hash: {source_commit_hash}"
                    )
                    logger.warning(
                        "Local repo is not upto date with remote. Updating local repo"
                    )
                    document_store = self._load_documents(paths, source_metadata)
                else:
                    logger.debug(
                        "Local repo is upto date with remote, loading from cache"
                    )
                    document_store = cache_document_store
            else:
                ignore_cache = ignore_cache or True
                logger.debug("No commit hash found in cache, loading fresh docs")
                document_store = self._load_documents(paths, source_metadata)
        else:
            logger.debug(f"{docstore_file} was not found, loading fresh docs")
            document_store = self._load_documents(paths, source_metadata)
        document_store = self.maybe_save_docstore(
            document_store, docstore_file, ignore_cache
        )
        return document_store

    def load_docstore(self, paths: BaseDataConfig = None) -> LlamaDocumentStore:
        if paths is None:
            paths = self.config.data_config
        if self.document_store is None:
            self.document_store = self.maybe_load_docstore(paths=paths)
        return self.document_store

    def create_vectorstore(
        self,
        documents: Dict[str, Union[LlamaDocument, BaseDocument]],
        metadata: Dict[str, str] = None,
    ):
        vector_store_kwargs = convert_llama_docstore_to_vectorstore_kwargs(documents)

        vectorstore = Chroma.from_documents(
            embedding=self.embedding_fn,
            collection_name=self.config.name,
            persist_directory=str(self.config.vectorstore_dir),
            collection_metadata=metadata,
            **vector_store_kwargs,
        )
        return vectorstore

    def maybe_load_vectorstore(
        self,
        docstore: LlamaDocumentStore,
    ):
        documents = docstore.docs
        metadata = docstore.ref_doc_info["metadata"]

        vectorstore_dir = pathlib.Path(self.config.vectorstore_dir)
        ignore_cache = self.config.data_config.ignore_cache

        if vectorstore_dir.is_dir() and not ignore_cache:
            logger.debug(f"{vectorstore_dir} was found, loading existing vector store")
            try:
                vectorstore = Chroma(
                    persist_directory=str(vectorstore_dir),
                    embedding_function=self.embedding_fn,
                    collection_name=self.config.name,
                    collection_metadata=metadata,
                )
            except:
                logger.warning(
                    f"Failed to load vector store from {vectorstore_dir}, recreating"
                )
                vectorstore = self.create_vectorstore(documents, metadata)
                ignore_cache = ignore_cache or True

            collection_ids = vectorstore._collection.get()["ids"]

            if not sorted(collection_ids) == sorted(documents.keys()):
                ignore_cache = ignore_cache or True
                logger.warning(
                    "The document ids in the vector store do not match the document ids loaded from files"
                )
                collection_docs_to_delete = set(collection_ids) - set(documents.keys())
                if collection_docs_to_delete:
                    logger.warning(
                        f"Deleting {len(collection_docs_to_delete)} documents from the vector store"
                    )
                    vectorstore._collection.delete(ids=list(collection_docs_to_delete))
                collection_docs_to_add = set(documents.keys()) - set(collection_ids)
                if collection_docs_to_add:
                    logger.debug(
                        f"Adding {len(collection_docs_to_add)} documents to the vector store"
                    )
                    vectorstore.add_documents(
                        [
                            documents[doc_id].to_langchain_format()
                            for doc_id in collection_docs_to_add
                        ],
                        ids=list(collection_docs_to_add),
                    )
            else:
                logger.debug(
                    "The document ids in the vector store match the document ids loaded from files"
                )
                ignore_cache = ignore_cache and False
        else:
            logger.debug(
                f"{vectorstore_dir} was not found, create a fresh vector store"
            )
            vectorstore = self.create_vectorstore(documents, metadata)
            ignore_cache = ignore_cache or True
        vectorstore = self.maybe_save_vectorstore(
            vectorstore, vectorstore_dir, ignore_cache
        )
        return vectorstore

    def load_vectorstore(self, paths: BaseDataConfig = None):
        if self.document_store is None:
            self.document_store = self.load_docstore(paths=paths)
        if self.vectorstore is None:
            self.vectorstore = self.maybe_load_vectorstore(self.document_store)
        return self.vectorstore

    def maybe_load_from_artifact(self, artifact_path: str = None, version="latest"):
        self.wandb_run = self.maybe_create_run()
        if artifact_path is None:
            artifact_path = f"{self.wandb_run.entity}/{self.wandb_run.project}/{self.config.name}:{version}"
        try:
            api = wandb.Api()
            artifact = api.artifact(artifact_path)
            metadata = artifact.metadata
            artifact_commit_hash = (
                metadata.get("doc_store_info").get("metadata").get("commit_hash")
            )
            source_metadata = self.load_docstore_metadata(self.config.data_config)
            source_commit_hash = source_metadata.get("commit_hash")
            return self.verify_and_load_artifact(
                artifact_path, artifact_commit_hash, source_commit_hash
            )
        except:
            logger.warning(
                f"Failed to load artifact from {artifact_path}, recreating artifact"
            )
            return self.save()


class CombinedDocStore(BaseDocStore):
    config: CombinedDocStoreConfig

    def __init__(self, config: CombinedDocStoreConfig = None, **kwargs):
        super().__init__(config, **kwargs)
        if config is not None:
            self.config = config
        self.document_store_dict = None

    @abc.abstractmethod
    def _load_docstore_dict(
        self, configs: Dict[str, DocStoreConfig]
    ) -> Dict[str, "BaseDocStore"]:
        raise NotImplementedError("You must implement this method in your subclass")

    def load_docstore_metadata(
        self, configs: Dict[str, DocStoreConfig]
    ) -> Dict[str, Any]:
        stores_metadata = {}
        for docstore_name, config in configs.items():
            logger.debug(f"Loading {docstore_name} from {config.cls}")
            docstore_class = load_docstore_class(sys.modules[__name__], config.cls)
            docstore: DocStore = docstore_class(config)
            stores_metadata[docstore_name] = docstore.load_docstore_metadata(
                config.data_config
            )

        metadata = {
            "stores_info": stores_metadata,
            "commit_hash": hashlib.md5(
                json.dumps(stores_metadata, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }

        return metadata

    def create_docstore(
        self,
        docstore_dict: Dict[str, DocStore],
        docstore_metadata: Dict[str, Any],
    ):
        logger.debug(f"Merging document stores stored")
        stores = []
        for docstore_name, docstore in docstore_dict.items():
            stores.append(docstore.document_store)
        self.document_store = LlamaDocumentStore.merge(stores)
        self.document_store.ref_doc_info["metadata"] = docstore_metadata
        return self.document_store

    def _load_docstore(
        self, configs: Dict[str, DocStoreConfig], metadata: Dict[str, Any]
    ):
        if self.document_store_dict is None:
            self.document_store_dict = self._load_docstore_dict(configs)
        return self.create_docstore(self.document_store_dict, metadata)

    def maybe_load_docstore(self, config: CombinedDocStoreConfig = None):

        docstore_file = pathlib.Path(config.docstore_file)
        ignore_cache = config.data_config.ignore_cache
        source_metadata = self.load_docstore_metadata(config.docstore_configs)

        if docstore_file.is_file() and not ignore_cache:
            logger.debug(f"{docstore_file} was found, loading existing docs")
            doc_store_dict = json.load(open(docstore_file))
            cache_document_store = LlamaDocumentStore.load_from_dict(doc_store_dict)
            cache_commit_hash = cache_document_store.ref_doc_info.get(
                "metadata", {}
            ).get("commit_hash")
            if cache_commit_hash is not None:
                source_commit_hash = source_metadata.get("commit_hash")
                if cache_commit_hash != source_commit_hash:
                    ignore_cache = ignore_cache or True
                    logger.debug(
                        f"cache_commit_hash: {cache_commit_hash}, source_commit_hash: {source_commit_hash}"
                    )
                    logger.warning(
                        "Local data is not upto date with source, recreating local data"
                    )
                    document_store = self._load_docstore(
                        config.docstore_configs, source_metadata
                    )
                else:
                    logger.debug(
                        "Local data is upto date with source, loading from cache"
                    )
                    document_store = cache_document_store
            else:
                ignore_cache = ignore_cache or True
                logger.debug("No commit hash found in cache, loading fresh docs")
                document_store = self._load_docstore(
                    config.docstore_configs, source_metadata
                )
        else:
            logger.debug(f"{docstore_file} was not found, loading fresh docs")
            document_store = self._load_docstore(
                config.docstore_configs, source_metadata
            )

        doc_store_dict = document_store.serialize_to_dict()
        if docstore_file.is_file() and ignore_cache:
            logger.debug(f"{docstore_file} was found, but ignoring cache")
            json.dump(
                doc_store_dict,
                open(docstore_file, "w"),
            )
        if not docstore_file.is_file():
            logger.debug(f"{docstore_file} was not found, writing to cache")
            docstore_file.parent.mkdir(parents=True, exist_ok=True)
            json.dump(
                doc_store_dict,
                open(docstore_file, "w"),
            )
        return document_store

    def load_docstore(self, config: CombinedDocStoreConfig = None):
        if config is None:
            config = self.config
        if self.document_store_dict is None:
            self.document_store_dict = self._load_docstore_dict(config.docstore_configs)

        if self.document_store is None:
            self.document_store = self.maybe_load_docstore(config)
        return self.document_store

    def create_vectorstore(self, document_store_dict, metadata: Dict[str, Any]):
        store_data = {}
        vectorstore = ChromaWithEmbeddings(
            collection_name=self.config.name,
            embedding_function=self.embedding_fn,
            persist_directory=str(self.config.vectorstore_dir),
            collection_metadata=metadata,
        )
        for name, docstore in document_store_dict.items():
            for key, values in docstore.vectorstore._collection.get(
                include=["documents", "metadatas", "embeddings"]
            ).items():
                store_data[key] = store_data.get(key, []) + values
        vectorstore.add_texts_and_embeddings(**store_data)
        return vectorstore

    def maybe_load_vectorstore(self, document_store_dict: Dict[str, DocStore]):
        vectorstore_dir = pathlib.Path(self.config.vectorstore_dir)
        ignore_cache = self.config.data_config.ignore_cache
        metadata = self.load_docstore_metadata(self.config.docstore_configs)
        if vectorstore_dir.is_dir() and not ignore_cache:
            logger.debug(f"{vectorstore_dir} was found, loading existing vector store")
            try:
                vectorstore = Chroma(
                    persist_directory=str(vectorstore_dir),
                    embedding_function=self.embedding_fn,
                    collection_name=self.config.name,
                    collection_metadata=metadata,
                )
            except:
                logger.warning(
                    f"Failed to load vector store from {vectorstore_dir}, recreating"
                )
                vectorstore = self.create_vectorstore(document_store_dict, metadata)
                ignore_cache = ignore_cache or True

            collection_ids = vectorstore._collection.get()["ids"]
            source_collection_ids = list()
            for name, docstore in document_store_dict.items():
                source_collection_ids += docstore.vectorstore._collection.get()["ids"]
            if not sorted(collection_ids) == sorted(source_collection_ids):
                logger.warning(
                    f"Vector store is not upto date with source, recreating vector store"
                )
                vectorstore = self.create_vectorstore(document_store_dict, metadata)
                ignore_cache = ignore_cache or True
            else:
                logger.debug(
                    "The document ids in the vector store match the document ids loaded from files"
                )
                ignore_cache = ignore_cache and False
        else:
            logger.debug(f"{vectorstore_dir} was not found, creating new vector store")
            vectorstore = self.create_vectorstore(document_store_dict, metadata)
            ignore_cache = ignore_cache or True
        vectorstore = self.maybe_save_vectorstore(
            vectorstore, vectorstore_dir, ignore_cache
        )
        return vectorstore

    def load_vectorstore(self, config: CombinedDocStoreConfig = None):
        if config is None:
            config = self.config
        if self.document_store is None:
            self.document_store = self.load_docstore(config)
        if self.vectorstore is None:
            self.vectorstore = self.maybe_load_vectorstore(
                self.document_store_dict,
            )
        return self.vectorstore

    def maybe_load_from_artifact(self, artifact_path: str = None, version="latest"):
        self.wandb_run = self.maybe_create_run()
        if artifact_path is None:
            artifact_path = f"{self.wandb_run.entity}/{self.wandb_run.project}/{self.config.name}:{version}"
        try:
            api = wandb.Api()
            artifact = api.artifact(artifact_path)
            metadata = artifact.metadata
            artifact_commit_hash = (
                metadata.get("doc_store_info").get("metadata").get("commit_hash")
            )
            source_metadata = self.load_docstore_metadata(self.config.docstore_configs)
            source_commit_hash = source_metadata.get("commit_hash")
            return self.verify_and_load_artifact(
                artifact_path, artifact_commit_hash, source_commit_hash
            )
        except:
            logger.warning(
                f"Failed to load artifact from {artifact_path}, recreating artifact"
            )
            return self.save()


class DocumentationDocStore(DocStore):
    config = DocumentationStoreConfig()

    def __init__(self, config: DocumentationStoreConfig = None, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(
        self, paths: BaseDataConfig, docstore_metadata: Dict[str, Any] = None
    ):
        local_paths = (paths.local_path / paths.base_path).rglob(paths.file_pattern)
        dir_name = paths.local_path.stem

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

        link_paths = map(lambda x: f"{paths.remote_path}{x}", link_paths)

        document_files = dict(zip(local_paths, link_paths))

        documents = []
        for f_name in tqdm(
            document_files, desc=f"Loading documentation from {paths.local_path}"
        ):
            try:
                documents.extend(UnstructuredMarkdownLoader(f_name).load())
            except:
                logger.warning(f"Failed to load documentation {f_name}")
        document_sections = self.md_text_splitter.split_documents(documents)

        document_store = self.create_docstore(
            document_sections, document_files, docstore_metadata
        )
        return document_store


class CodeDocStore(DocStore):
    config = CodeStoreConfig()

    def __init__(self, config: CodeStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(
        self, paths: BaseDataConfig, docstore_metadata: Dict[str, Any] = None
    ):
        local_paths = (paths.local_path / paths.base_path).rglob(paths.file_pattern)

        document_files = map_local_to_remote(
            local_paths, paths.local_path.stem, paths.remote_path
        )

        documents = []
        for f_name in tqdm(
            document_files, desc=f"Loading code from {paths.local_path}"
        ):
            try:
                if paths.file_pattern == "*.ipynb":
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
        document_store = self.create_docstore(
            document_sections, document_files, docstore_metadata
        )
        return document_store


class ExamplesCodeDocStore(CodeDocStore):
    config = ExamplesCodeStoreConfig()

    def __init__(self, config: ExamplesCodeStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(
        self, paths: BaseDataConfig, docstore_metadata: Dict[str, Any] = None
    ):
        return super()._load_documents(paths, docstore_metadata)


class ExamplesNotebookDocStore(CodeDocStore):
    config = ExamplesNotebookStoreConfig()

    def __init__(self, config: ExamplesNotebookStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(
        self, paths: BaseDataConfig, docstore_metadata: Dict[str, Any] = None
    ):
        return super()._load_documents(paths, docstore_metadata)


class SDKCodeDocStore(CodeDocStore):
    config = SDKCodeStoreConfig()

    def __init__(self, config: SDKCodeStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(
        self, paths: BaseDataConfig, docstore_metadata: Dict[str, Any] = None
    ):
        return super()._load_documents(paths, docstore_metadata)


class CsvDocStore(DocStore):
    config = CsvStoreConfig()

    def __init__(self, config: CsvStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths, docstore_metadata: Dict[str, Any] = None):
        csv_paths = (paths.local_path / paths.base_path).rglob(paths.file_pattern)
        all_documents = []
        for path in csv_paths:
            df = pd.read_csv(path).fillna("")
            if "title" in df.columns:
                df["question"] = df["title"] + "\n\n" + df["question"]
            if "source" not in df.columns:
                df["source"] = f"{str(path)}-" + df.index.map(str)
            df["source"] = df.apply(
                lambda x: f"{path.stem}-" + str(x.name)
                if not x["source"]
                else x["source"],
                axis=1,
            )
            data = df.apply(
                lambda x: f"Question:\n{'-' * 10}\n{x['question']}\n\nAnswer:\n{'-' * 10}\n{x['answer']}",
                axis=1,
            )

            data = pd.DataFrame(data, columns=["reference"])
            data["source"] = df["source"]
            documents = data.to_dict(orient="records")
            documents = [
                Document(
                    page_content=doc["reference"], metadata={"source": doc["source"]}
                )
                for doc in tqdm(documents, desc=f"loading csv data from {path}")
            ]
            all_documents.extend(documents)
        document_sections = self.md_text_splitter.split_documents(all_documents)
        document_store = self.create_docstore(
            document_sections, None, docstore_metadata
        )
        return document_store


class JsonlDocStore(DocStore):
    config = JsonlStoreConfig()

    def __init__(self, config: JsonlStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths, docstore_metadata: Dict[str, Any] = None):

        jsonl_paths = (paths.local_path / paths.base_path).rglob(paths.file_pattern)

        all_documents = []
        for path in jsonl_paths:
            for line in path.open("r"):
                doc = json.loads(line)
                document = Document(
                    page_content=doc["document"], metadata={"source": doc["source"]}
                )
                all_documents.append(document)
        document_sections = self.md_text_splitter.split_documents(all_documents)
        document_store = self.create_docstore(
            document_sections, None, docstore_metadata
        )
        return document_store


class ExtraDataDocStore(JsonlDocStore):
    config = ExtraDataStoreConfig()

    def __init__(self, config: ExtraDataStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths, docstore_metadata: Dict[str, Any] = None):
        return super()._load_documents(paths, docstore_metadata)


class WandbotDocStore(CombinedDocStore):
    config: WandbotDocStoreConfig = WandbotDocStoreConfig()

    def __init__(self, config: WandbotDocStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_docstore_dict(
        self, configs: Dict[str, DocStoreConfig]
    ) -> Dict[str, DocStore]:
        docstores = {}
        for docstore_name, config in configs.items():
            logger.debug(f"Loading {docstore_name} from {config.cls}")
            docstore_class = getattr(sys.modules[__name__], config.cls)
            docstore = docstore_class(config)
            docstores[docstore_name] = docstore.load()
        return docstores
