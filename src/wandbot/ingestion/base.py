import abc
import hashlib
import json
import logging
import pathlib
from typing import List, Dict, Optional, Any, Union

import tiktoken
import wandb
from langchain import LLMChain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chat_models import ChatOpenAI
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

from src.wandbot.ingestion.settings import (
    DocStoreConfig,
    BaseDataConfig,
    CombinedDocStoreConfig,
)
from src.wandbot.ingestion.utils import (
    fetch_git_remote_hash,
    md5_dir,
    ChromaWithEmbeddings,
)
from src.wandbot.prompts import load_hyde_prompt

logger = logging.getLogger(__name__)


def add_metadata(
    documents: List[Document], source_map: Optional[Dict[str, str]] = None
) -> List[Document]:
    out_documents = []
    for document in documents:
        doc_id = hashlib.md5(document.page_content.encode("UTF-8")).hexdigest()
        if isinstance(source_map, dict):
            source = source_map.get(
                document.metadata["source"], document.metadata["source"]
            )
        else:
            source = document.metadata["source"]
        metadata = {"source": source, "doc_id": doc_id}
        out_documents.append(
            Document(page_content=document.page_content, metadata=metadata)
        )
    return out_documents


def convert_llama_docstore_to_vectorstore_kwargs(
    documents: Dict[str, LlamaDocument]
) -> Dict[str, Any]:
    docs_dict = dict()
    for doc_id, document in documents.items():
        docs_dict["ids"] = docs_dict.get("ids", []) + [doc_id]
        docs_dict["documents"] = docs_dict.get("documents", []) + [
            document.to_langchain_format()
        ]
    return docs_dict


class BaseDocStore:
    config = DocStoreConfig()

    def __init__(self, config: Optional[DocStoreConfig] = None, **kwargs):

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
        self.document_store: Optional[LlamaDocumentStore] = None

        self.hyde_prompt = load_hyde_prompt(self.config.hyde_prompt)

        self.embedding_fn = HypotheticalDocumentEmbedder(
            llm_chain=LLMChain(
                llm=ChatOpenAI(temperature=self.config.hyde_temperature),
                prompt=self.hyde_prompt,
            ),
            base_embeddings=OpenAIEmbeddings(),
        )
        self.vector_store = None
        self.wandb_run = None

    def _maybe_create_run(
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

    def _make_documents_tokenization_safe(self, documents):
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

    def create_docstore(
        self,
        documents: List[Document],
        source_map: Optional[Dict[str, str]] = None,
        metadata: Dict[str, str] = None,
    ) -> LlamaDocumentStore:
        documents = self._make_documents_tokenization_safe(documents)
        documents = self.token_splitter.split_documents(documents)
        documents = add_metadata(documents, source_map)
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
    def _load_documents(self, paths: BaseDataConfig):
        raise NotImplementedError(
            "You need to implement this method in the subclass to parse and load your documents"
        )

    def _maybe_load_documents(self, paths=None):
        docstore_file = pathlib.Path(self.config.docstore_file)
        ignore_cache = self.config.data_config.ignore_cache
        if self.document_store is None:
            if docstore_file.is_file() and not ignore_cache:
                logger.debug(f"{docstore_file} was found, loading existing docs")
                doc_store_dict = json.load(open(docstore_file))
                cache_document_store = LlamaDocumentStore.load_from_dict(doc_store_dict)
                cache_commit_hash = cache_document_store.ref_doc_info.get(
                    "metadata", {}
                ).get("commit_hash")
                if self.config.data_config.is_git_repo:
                    source_commit_hash = fetch_git_remote_hash(
                        paths.repo_path, self.config.data_config.git_id_file
                    )
                else:
                    logger.debug(
                        "Docstore is not sourced from a git repo, validating with commit hash of the directory instead."
                    )
                    source_commit_hash = md5_dir(
                        self.config.data_config.local_path,
                        self.config.data_config.file_pattern,
                    )
                if cache_commit_hash != source_commit_hash:
                    ignore_cache = True
                    logger.debug(
                        f"cache_commit_hash: {cache_commit_hash}, source_commit_hash: {source_commit_hash}"
                    )
                    logger.warning(
                        "Local repo is not upto date with remote. Updating local repo"
                    )
                    document_store = self._load_documents(paths)
                else:
                    logger.debug(
                        "Local repo is upto date with remote, loading from cache"
                    )
                    document_store = cache_document_store
            else:
                logger.debug(f"{docstore_file} was not found, loading fresh docs")
                document_store = self._load_documents(paths)

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
            self.document_store = document_store
        return self.document_store

    def load_docstore(self, paths: BaseDataConfig = None) -> LlamaDocumentStore:
        if paths is None:
            paths = self.config.data_config
        if self.document_store is None:
            self.document_store = self._maybe_load_documents(paths=paths)
        return self.document_store

    def _create_vector_store(self, documents: Dict[str, LlamaDocument]):
        if self.vector_store is None:
            vector_store_kwargs = convert_llama_docstore_to_vectorstore_kwargs(
                documents
            )

            self.vector_store = Chroma.from_documents(
                embedding=self.embedding_fn,
                collection_name=self.config.name,
                persist_directory=str(self.config.vectorstore_dir),
                **vector_store_kwargs,
            )
        return self.vector_store

    def maybe_load_vector_store(
        self, documents: Dict[str, Union[LlamaDocument, BaseDocument]]
    ):
        vector_store_dir = pathlib.Path(self.config.vectorstore_dir)
        ignore_cache = self.config.data_config.ignore_cache
        if self.vector_store is None:
            if vector_store_dir.is_dir() and not ignore_cache:
                logger.debug(
                    f"{vector_store_dir} was found, loading existing vector store"
                )
                try:
                    vector_store = Chroma(
                        persist_directory=str(vector_store_dir),
                        embedding_function=self.embedding_fn,
                        collection_name=self.config.name,
                    )
                except:
                    logger.warning(
                        f"Failed to load vector store from {vector_store_dir}, recreating"
                    )
                    vector_store = self._create_vector_store(documents)
                    ignore_cache = ignore_cache or True

                collection_ids = vector_store._collection.get()["ids"]

                if not sorted(collection_ids) == sorted(documents.keys()):
                    ignore_cache = ignore_cache or True
                    logger.warning(
                        "The document ids in the vector store do not match the document ids loaded from files"
                    )
                    collection_docs_to_delete = set(collection_ids) - set(
                        documents.keys()
                    )
                    if collection_docs_to_delete:
                        logger.warning(
                            f"Deleting {len(collection_docs_to_delete)} documents from the vector store"
                        )
                        vector_store._collection.delete(
                            ids=list(collection_docs_to_delete)
                        )
                    collection_docs_to_add = set(documents.keys()) - set(collection_ids)
                    if collection_docs_to_add:
                        logger.debug(
                            f"Adding {len(collection_docs_to_add)} documents to the vector store"
                        )
                        vector_store.add_documents(
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
                    f"{vector_store_dir} was not found, create a fresh vector store"
                )
                vector_store = self._create_vector_store(documents)
                ignore_cache = ignore_cache or True
            if vector_store_dir.exists() and ignore_cache:
                logger.debug(f"{vector_store_dir} was found, but overwriting cache")
                vector_store.persist()
            elif not vector_store_dir.exists():
                logger.debug(f"{vector_store_dir} was not found, writing to cache")
                vector_store.persist()
            self.vector_store = vector_store
        return self.vector_store

    def load_vector_store(self, documents=None, paths=None):
        if documents is None:
            documents = self.load_docstore(paths=paths)
        if self.vector_store is None:
            self.vector_store = self.maybe_load_vector_store(documents.docs)
        return self.vector_store

    def save(self):
        self.wandb_run = self._maybe_create_run()
        if self.document_store is None:
            self.document_store = self.load_docstore(self.config.data_config)
        if self.vector_store is None:
            self.vector_store = self.load_vector_store()
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

    def _load_from_artifact(self, artifact_path: str):
        artifact = self.wandb_run.use_artifact(artifact_path)
        artifact_dir = artifact.download()
        artifact_dir = pathlib.Path(artifact_dir)

        self.config.docstore_file = artifact_dir / self.config.docstore_file.name
        self.config.vectorstore_dir = artifact_dir / self.config.vectorstore_dir.name
        self.hyde_prompt = load_hyde_prompt(
            str(artifact_dir / self.config.hyde_prompt.name)
        )
        self.document_store = self.load_docstore(self.config.data_config)
        self.vector_store = self.load_vector_store()
        return self

    def maybe_load_from_artifact(self, artifact_path: str = None, version="latest"):
        self.wandb_run = self._maybe_create_run()
        if artifact_path is None:
            artifact_path = f"{self.wandb_run.entity}/{self.wandb_run.project}/{self.config.name}:{version}"
        try:
            api = wandb.Api()
            artifact = api.artifact(artifact_path)
            metadata = artifact.metadata
            artifact_commit_hash = (
                metadata.get("doc_store_info").get("metadata").get("commit_hash")
            )
            if artifact_commit_hash is not None:
                if self.config.data_config.is_git_repo:
                    source_commit_hash = fetch_git_remote_hash(
                        self.config.data_config.repo_path,
                        self.config.data_config.git_id_file,
                    )
                else:
                    logger.debug(
                        "Docstore is not sourced from a git repo, validating with commit hash of the directory "
                        "and files instead."
                    )
                    source_commit_hash = md5_dir(
                        self.config.data_config.local_path,
                        self.config.data_config.file_pattern,
                    )
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
                    return self._load_from_artifact(artifact_path)
            else:
                logger.debug(f"Artifact commit hash not found, recreating artifact")
                return self.save()
        except:
            logger.warning(
                f"Failed to load artifact from {artifact_path}, recreating artifact"
            )
            return self.save()

    def load(self, artifact_path: str = None, version="latest"):
        return self.maybe_load_from_artifact(artifact_path, version=version)


class BaseCombinedDocStore:
    config: CombinedDocStoreConfig

    def __init__(self, config: CombinedDocStoreConfig = None, **kwargs):
        if config is not None:
            self.config = config
        self.document_store_dict = None
        self.document_store: Optional[LlamaDocumentStore] = None
        self.vector_store = None
        self.wandb_run = None
        self.hyde_prompt = load_hyde_prompt(self.config.hyde_prompt)
        self.embedding_fn = HypotheticalDocumentEmbedder(
            llm_chain=LLMChain(
                llm=ChatOpenAI(temperature=self.config.hyde_temperature),
                prompt=self.hyde_prompt,
            ),
            base_embeddings=OpenAIEmbeddings(),
        )

    def _maybe_create_run(self):
        if self.wandb_run is None:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project, entity=self.config.wandb_entity
            )
        return self.wandb_run

    def load_docstore_dict(self, configs: Dict[str, DocStoreConfig] = None):
        self.document_store_dict = self._load_docstore_dict(configs)
        return self.document_store_dict

    @abc.abstractmethod
    def _load_docstore_dict(self, configs: Dict[str, DocStoreConfig]):
        raise NotImplementedError("You need to implement this method is the subclass")

    def create_docstore(self, config: CombinedDocStoreConfig):
        logger.debug(f"Creating document store from {config.docstore_configs.keys()}")
        self.document_store_dict = self.load_docstore_dict(config.docstore_configs)
        stores = []
        stores_metadata = {}
        for k, v in self.document_store_dict.items():
            v.document_store.ref_doc_info["metadata"]["source"] = v.config.name
            v.document_store.ref_doc_info["metadata"]["cls"] = v.config.cls
            stores.append(v.document_store)
            stores_metadata[k] = v.document_store.ref_doc_info["metadata"]

        metadata = {"stores_info": stores_metadata}
        metadata["commit_hash"] = hashlib.md5(
            json.dumps(stores_metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()

        logger.debug(f"Merging document stores stored")
        self.document_store = LlamaDocumentStore.merge(stores)
        self.document_store.ref_doc_info["metadata"] = metadata
        return self.document_store

    def maybe_load_docstore(self, config: CombinedDocStoreConfig = None):
        docstore_file = pathlib.Path(config.docstore_file)
        ignore_cache = config.data_config.ignore_cache
        if self.document_store is None:
            if docstore_file.is_file() and not ignore_cache:
                logger.debug(f"{docstore_file} was found, loading existing docs")
                doc_store_dict = json.load(open(docstore_file))
                cache_document_store = LlamaDocumentStore.load_from_dict(doc_store_dict)

    def load_docstore(self, config: CombinedDocStoreConfig = None):
        if config is None:
            config = self.config
        if self.document_store is None:
            self.document_store = self.create_docstore(config)
        return self.document_store

    def create_vector_store(self, config: CombinedDocStoreConfig):
        self.document_store_dict = self.load_docstore_dict(config.docstore_configs)
        store_data = {}
        vector_store = ChromaWithEmbeddings(
            collection_name=self.config.name,
            embedding_function=self.embedding_fn,
            persist_directory=str(self.config.vectorstore_dir),
        )
        for name, docstore in self.document_store_dict.items():
            for key, values in docstore.vectorstore._collection.get().items():
                store_data[key] = store_data.get(key, []) + values
        vector_store.add_texts_and_embeddings(**store_data)
        return vector_store

    def load_vector_store(self, config: CombinedDocStoreConfig = None):
        if config is None:
            configs = self.config
        if self.vector_store is None:
            self.vector_store = self.create_vector_store(config)
        return self.vector_store
