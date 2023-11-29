"""
This module contains classes and functions for preparing data in the Wandbot ingestion system.

The module includes the following classes:
- `DataLoader`: A base class for data loaders that provides a base implementation for lazy loading of documents.
- `DocodileDataLoader`: A data loader specifically designed for Docodile documents.
- `CodeDataLoader`: A data loader for code documents.

The module also includes the following functions:
- `load`: Loads and prepares data for the Wandbot ingestion system.

Typical usage example:

    load(project="my_project", entity="my_entity", result_artifact_name="raw_dataset")
"""

import json
import os
import pathlib
from typing import Iterator
from urllib.parse import urljoin

import markdown
import nbformat
from langchain.document_loaders import TextLoader
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from nbconvert import MarkdownExporter

import wandb
from wandbot.ingestion.config import (
    DataStoreConfig,
    DocodileEnglishStoreConfig,
    DocodileJapaneseStoreConfig,
    ExampleCodeStoreConfig,
    ExampleNotebookStoreConfig,
)
from wandbot.ingestion.utils import (
    EXTENSION_MAP,
    clean_contents,
    fetch_git_repo,
)
from wandbot.utils import get_logger

logger = get_logger(__name__)


class DataLoader(BaseLoader):
    """A base class for data loaders.

    This class provides a base implementation for lazy loading of documents.
    Subclasses should implement the `lazy_load` method to define the specific
    loading behavior.
    """

    def __init__(self, config: DataStoreConfig):
        """Initializes the DataLoader instance.

        Args:
            config: The configuration for the data store.
        """
        self.config = config
        self.metadata = None

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Documents.

        This method should be implemented by subclasses to define the specific
        loading behavior.

        Returns:
            An iterator of Document objects.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )

    def load(self):
        """Loads the documents.

        Returns:
            A list of Document objects.
        """
        documents = list(self.lazy_load())
        self.metadata.update({"num_documents": len(documents)})
        return documents

    def _get_local_paths(self):
        if self.config.data_source.is_git_repo:
            self.metadata = fetch_git_repo(
                self.config.data_source, self.config.data_source.git_id_file
            )

        local_paths = []
        file_patterns = (
            [self.config.data_source.file_pattern]
            if isinstance(self.config.data_source.file_pattern, str)
            else self.config.data_source.file_pattern
        )
        for file_pattern in file_patterns:
            local_paths.extend(
                list(
                    (
                        self.config.data_source.local_path
                        / self.config.data_source.base_path
                    ).rglob(file_pattern)
                )
            )
        return local_paths


class DocodileDataLoader(DataLoader):
    """A data loader for Docodile documents.

    This class provides a data loader specifically designed for Docodile documents.
    It implements the lazy_load method to define the loading behavior.

    Attributes:
        config: The configuration for the data store.
    """

    @staticmethod
    def extract_slug(file_path: pathlib.Path) -> str:
        """Extracts the slug from a file.

        Args:
            file_path: The path to the file.

        Returns:
            The extracted slug.
        """
        with open(file_path, "r") as file:
            content = file.read()
            md = markdown.Markdown(extensions=["meta"])
            md.convert(content)
            meta = md.Meta.get("slug", [""])
            return meta[0]

    def generate_site_url(
        self, base_path: pathlib.Path, file_path: pathlib.Path
    ) -> str:
        """Generates the site URL for a file.

        Args:
            base_path: The base path of the file.
            file_path: The path to the file.

        Returns:
            The generated site URL.
        """
        relative_path = file_path.relative_to(base_path)
        if relative_path.parts[0] == "guides":
            chapter = "guides"
            slug = self.extract_slug((base_path / "guides") / "intro.md")
            file_loc = file_path.relative_to((base_path / "guides")).parent
        elif relative_path.parts[0] == "ref":
            chapter = "ref"
            slug = self.extract_slug((base_path / "ref") / "README.md")
            file_loc = file_path.relative_to((base_path / "ref")).parent
        elif relative_path.parts[0] == "tutorials":
            chapter = "tutorials"
            slug = self.extract_slug(
                (base_path / "tutorials") / "intro_to_tutorials.md"
            )
            file_loc = file_path.relative_to((base_path / "tutorials")).parent
        else:
            chapter = ""
            slug = ""
            file_loc = ""

        file_name = file_path.stem
        if file_path.name in ("intro.md", "README.md", "intro_to_tutorials.md"):
            file_name = ""
        site_relative_path = os.path.join(chapter, slug, file_loc, file_name)
        site_url = urljoin(
            str(self.config.data_source.remote_path), str(site_relative_path)
        )
        if "other/" in site_url:
            site_url = site_url.replace("other/", "")

        return site_url

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Docodile documents.

        This method implements the lazy loading behavior for Docodile documents.

        Yields:
            A Document object.
        """
        local_paths = self._get_local_paths()
        document_files = {
            local_path: self.generate_site_url(
                self.config.data_source.local_path
                / self.config.data_source.base_path,
                local_path,
            )
            for local_path in local_paths
        }

        for f_name in document_files:
            try:
                document = TextLoader(f_name).load()[0]
                contents = document.page_content
                document.page_content = clean_contents(contents)
                document.metadata["file_type"] = os.path.splitext(
                    document.metadata["source"]
                )[-1]
                document.metadata["source"] = document_files[
                    document.metadata["source"]
                ]
                document.metadata["language"] = self.config.language
                yield document
            except Exception as e:
                logger.warning(
                    f"Failed to load documentation {f_name} due to {e}"
                )


class CodeDataLoader(DataLoader):
    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for code documents.

        This method implements the lazy loading behavior for code documents.

        Yields:
            A Document object.
        """
        local_paths = self._get_local_paths()

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

        for f_name in document_files:
            try:
                if os.path.splitext(f_name)[-1] == ".ipynb":
                    document = TextLoader(f_name).load()[0]
                    contents = document.page_content
                    notebook = nbformat.reads(contents, as_version=4)
                    md_exporter = MarkdownExporter(template="classic")
                    (body, resources) = md_exporter.from_notebook_node(notebook)
                    cleaned_body = clean_contents(body)
                    document.page_content = cleaned_body
                else:
                    document = TextLoader(f_name).load()[0]
                document.metadata["file_type"] = os.path.splitext(
                    document.metadata["source"]
                )[-1]
                document.metadata["source"] = document_files[
                    document.metadata["source"]
                ]
                document.metadata["language"] = EXTENSION_MAP[
                    document.metadata["file_type"]
                ]
                yield document
            except Exception as e:
                logger.warning(
                    f"Failed to load code in {f_name} with error {e}"
                )


def load(
    project: str,
    entity: str,
    result_artifact_name: str = "raw_dataset",
) -> str:
    """Load and prepare data for the Wandbot ingestion system.

    This function initializes a Wandb run, creates an artifact for the prepared dataset,
    and loads and prepares data from different loaders. The prepared data is then saved
    in the docstore directory and added to the artifact.

    Args:
        project: The name of the Wandb project.
        entity: The name of the Wandb entity.
        result_artifact_name: The name of the result artifact. Default is "raw_dataset".

    Returns:
        The latest version of the prepared dataset artifact in the format
        "{entity}/{project}/{result_artifact_name}:latest".
    """
    run = wandb.init(project=project, entity=entity, job_type="prepare_dataset")
    artifact = wandb.Artifact(
        result_artifact_name,
        type="dataset",
        description="Raw documents for wandbot",
    )

    en_docodile_loader = DocodileDataLoader(DocodileEnglishStoreConfig())
    ja_docodile_loader = DocodileDataLoader(DocodileJapaneseStoreConfig())
    examples_code_loader = CodeDataLoader(ExampleCodeStoreConfig())
    examples_notebook_loader = CodeDataLoader(ExampleNotebookStoreConfig())

    for loader in [
        en_docodile_loader,
        ja_docodile_loader,
        examples_code_loader,
        examples_notebook_loader,
    ]:
        loader.config.docstore_dir.mkdir(parents=True, exist_ok=True)

        with (loader.config.docstore_dir / "config.json").open("w") as f:
            f.write(loader.config.model_dump_json())

        with (loader.config.docstore_dir / "documents.jsonl").open("w") as f:
            for document in loader.load():
                document_json = {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                f.write(json.dumps(document_json) + "\n")
        with (loader.config.docstore_dir / "metadata.json").open("w") as f:
            json.dump(loader.metadata, f)

        artifact.add_dir(
            str(loader.config.docstore_dir),
            name=loader.config.docstore_dir.name,
        )
    run.log_artifact(artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"
