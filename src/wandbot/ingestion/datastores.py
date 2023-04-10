import logging
import pathlib
from types import SimpleNamespace

import pandas as pd
from langchain.document_loaders import UnstructuredMarkdownLoader, NotebookLoader
from langchain.schema import Document
from tqdm import tqdm

from src.wandbot.ingestion.base import DocStore, add_metadata
from src.wandbot.ingestion.settings import DocumentationStoreConfig, BaseDataConfig
from src.wandbot.ingestion.utils import fetch_git_repo, map_local_to_remote

logger = logging.getLogger(__name__)


class DocumentationDocStore(DocStore):
    config = DocumentationStoreConfig()

    def __init__(self, config: DocumentationStoreConfig = None, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths: BaseDataConfig):
        if paths.is_git_repo:
            doc_store_metadata = fetch_git_repo(paths)
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


class CodeDocStore(DocStore):
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)

    def _load_documents(self, paths, base_path="", file_pattern="*.py"):
        repo_metadata = fetch_git_repo(paths)
        code_paths = (paths.local / base_path).rglob(file_pattern)

        code_files = map_local_to_remote(
            code_paths, paths.local_path.stem, paths.remote_path
        )

        codes = []
        for f_name in tqdm(code_files, desc=f"Loading code from {paths.local}"):
            try:
                if file_pattern == "*.ipynb":
                    codes.extend(
                        NotebookLoader(
                            f_name,
                            include_outputs=False,
                            max_output_length=0,
                            remove_newline=True,
                        ).load()
                    )
                else:
                    contents = open(f_name, "r").read()
                    codes.append(
                        Document(page_content=contents, metadata={"source": f_name})
                    )
            except:
                logger.warning(f"Failed to load code in {f_name}")
        code_sections = self.code_text_splitter.split_documents(codes)
        code_sections = self.token_splitter.split_documents(code_sections)
        code_sections = add_metadata(code_sections, code_files)
        return {"documents": code_sections, "metadata": repo_metadata}


class ExamplesCodeDocStore(CodeDocStore):
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.config.data_paths = (
            SimpleNamespace(
                local=pathlib.Path("../data/raw_dataset/examples"),
                remote="https://github.com/wandb/examples/blob/master/",
                repo_path="https://github.com/wandb/examples",
            ),
        )

    def _load_documents(self, paths, base_path="examples", file_pattern="*.py"):
        return super()._load_documents(
            paths, base_path=base_path, file_pattern=file_pattern
        )


class ExamplesNotebookDocStore(CodeDocStore):
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.config.data_paths = (
            SimpleNamespace(
                local=pathlib.Path("../data/raw_dataset/examples"),
                remote="https://github.com/wandb/examples/blob/master/",
                repo_path="https://github.com/wandb/examples",
            ),
        )

    def _load_documents(self, paths, base_path="colabs", file_pattern="*.ipynb"):
        return super()._load_documents(
            paths, base_path=base_path, file_pattern=file_pattern
        )


class SdkCodeDocStore(CodeDocStore):
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.config.data_paths = (
            SimpleNamespace(
                local=pathlib.Path("../data/raw_dataset/wandb"),
                remote="https://github.com/wandb/wandb/blob/main/",
                repo_path="https://github.com/wandb/wandb",
            ),
        )

    def _load_documents(self, paths, base_path="wandb", file_pattern="*.py"):
        return super()._load_documents(
            paths, base_path=base_path, file_pattern=file_pattern
        )


class CsvDocStore(DocStore):
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.config.data_paths = SimpleNamespace(
            local=pathlib.Path("../data/raw_dataset/extra_data"),
            remote=None,
            repo_path=None,
        )

    def _load_documents(self, paths, base_path="", file_pattern="*.csv"):
        csv_paths = (paths.local / base_path).rglob(file_pattern)
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

            document_sections = self.md_text_splitter.split_documents(documents)
            document_sections = self.token_splitter.split_documents(document_sections)
            document_sections = add_metadata(document_sections, None)
            all_documents.extend(document_sections)
        return all_documents


def main():
    DocumentationDocStore().load()
