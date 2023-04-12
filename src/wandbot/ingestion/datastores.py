import json
import logging
import pathlib
import sys
from typing import Dict

import pandas as pd
from langchain.document_loaders import (
    UnstructuredMarkdownLoader,
    NotebookLoader,
    TextLoader,
)
from langchain.schema import Document
from tqdm import tqdm

from src.wandbot.ingestion.base import BaseDocStore, BaseCombinedDocStore
from src.wandbot.ingestion.settings import (
    DocumentationStoreConfig,
    BaseDataConfig,
    CodeStoreConfig,
    ExamplesCodeStoreConfig,
    ExamplesNotebookStoreConfig,
    SDKCodeStoreConfig,
    CsvStoreConfig,
    JsonlStoreConfig,
    ExtraDataStoreConfig,
    WandbotDocStoreConfig,
    DocStoreConfig,
)
from src.wandbot.ingestion.utils import fetch_git_repo, map_local_to_remote, md5_dir

logger = logging.getLogger(__name__)


class DocumentationDocStore(BaseDocStore):
    config = DocumentationStoreConfig()

    def __init__(self, config: DocumentationStoreConfig = None, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths: BaseDataConfig):
        if paths.is_git_repo:
            doc_store_metadata = fetch_git_repo(
                paths, self.config.data_config.git_id_file
            )
        else:
            doc_store_metadata = {}
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
            document_sections, document_files, doc_store_metadata
        )
        return document_store


class CodeDocStore(BaseDocStore):
    config = CodeStoreConfig()

    def __init__(self, config: CodeStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths: BaseDataConfig):
        if paths.is_git_repo:
            doc_store_metadata = fetch_git_repo(
                paths, self.config.data_config.git_id_file
            )
        else:
            doc_store_metadata = {}
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
            document_sections, document_files, doc_store_metadata
        )
        return document_store


class ExamplesCodeDocStore(CodeDocStore):
    config = ExamplesCodeStoreConfig()

    def __init__(self, config: ExamplesCodeStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths: BaseDataConfig):
        return super()._load_documents(paths)


class ExamplesNotebookDocStore(CodeDocStore):
    config = ExamplesNotebookStoreConfig()

    def __init__(self, config: ExamplesNotebookStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths: BaseDataConfig):
        return super()._load_documents(paths)


class SDKCodeDocStore(CodeDocStore):
    config = SDKCodeStoreConfig()

    def __init__(self, config: SDKCodeStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths: BaseDataConfig):
        return super()._load_documents(
            paths,
        )


class CsvDocStore(BaseDocStore):
    config = CsvStoreConfig()

    def __init__(self, config: CsvStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths):
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
            document_sections, None, doc_store_metadata
        )
        return document_store


class JsonlDocStore(BaseDocStore):
    config = JsonlStoreConfig()

    def __init__(self, config: JsonlStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths):
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
            document_sections, None, doc_store_metadata
        )
        return document_store


class ExtraDataDocStore(JsonlDocStore):
    config = ExtraDataStoreConfig()

    def __init__(self, config: ExtraDataStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths):
        return super()._load_documents(paths)


class WandbotDocStore(BaseCombinedDocStore):
    config: WandbotDocStoreConfig = WandbotDocStoreConfig()

    def __init__(self, config: WandbotDocStoreConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _load_docstore_dict(self, configs: Dict[str, DocStoreConfig]):
        docstores = {}
        for docstore_name, config in configs.items():
            logger.debug(f"Loading {docstore_name} from {config.cls}")
            docstore_class = getattr(sys.modules[__name__], config.cls)
            docstore = docstore_class(config)
            docstores[docstore_name] = docstore.load()
        return docstores
