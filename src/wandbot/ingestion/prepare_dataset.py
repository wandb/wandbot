import hashlib
import json
import os
from typing import Iterator
from urllib.parse import urljoin

import markdown
import wandb
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from wandbot.ingestion.config import (
    DataStoreConfig,
    DocodileEnglishStoreConfig,
    DocodileJapaneseStoreConfig,
    ExampleCodeStoreConfig,
    ExampleNotebookStoreConfig,
    SDKCodeStoreConfig,
)
from wandbot.ingestion.utils import EXTENSION_MAP, WandbNotebookLoader, fetch_git_repo
from wandbot.utils import get_logger

logger = get_logger(__name__)


class DataLoader(BaseLoader):
    def __init__(self, config: DataStoreConfig):
        self.config = config
        self.metadata = None

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Documents."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )

    def load(self):
        documents = list(self.lazy_load())
        self.metadata.update({"num_documents": len(documents)})
        return documents


class DocodileDataLoader(DataLoader):
    def extract_slug(self, file_path):
        with open(file_path, "r") as file:
            content = file.read()
            md = markdown.Markdown(extensions=["meta"])
            md.convert(content)
            meta = md.Meta.get("slug", [""])
            return meta[0]

    def generate_site_url(self, base_path, file_path):
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
        document_files = {
            local_path: self.generate_site_url(
                self.config.data_source.local_path / self.config.data_source.base_path,
                local_path,
            )
            for local_path in local_paths
        }

        for f_name in document_files:
            try:
                document = UnstructuredMarkdownLoader(f_name).load()[0]
                document.metadata["hash"] = hashlib.md5(
                    (
                        str(document.metadata["source"]) + str(document.page_content)
                    ).encode("UTF-8")
                ).hexdigest()
                document.metadata["file_type"] = os.path.splitext(
                    document.metadata["source"]
                )[-1]
                document.metadata["source"] = document_files[
                    document.metadata["source"]
                ]
                document.metadata["source_metadata"] = json.dumps(self.metadata)
                document.metadata["language"] = self.config.language
                yield document
            except Exception as e:
                logger.warning(f"Failed to load documentation {f_name} due to {e}")


class CodeDataLoader(DataLoader):
    def lazy_load(self) -> Iterator[Document]:
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
                if self.config.data_source.file_pattern == "*.ipynb":
                    document = WandbNotebookLoader(
                        f_name,
                        include_outputs=False,
                        max_output_length=0,
                        remove_newline=True,
                    ).load()[0]
                else:
                    document = TextLoader(f_name).load()[0]
                document.metadata["hash"] = hashlib.md5(
                    (
                        str(document.metadata["source"]) + str(document.page_content)
                    ).encode("UTF-8")
                ).hexdigest()
                document.metadata["file_type"] = os.path.splitext(
                    document.metadata["source"]
                )[-1]
                document.metadata["source"] = document_files[
                    document.metadata["source"]
                ]
                document.metadata["source_metadata"] = json.dumps(self.metadata)
                document.metadata["language"] = EXTENSION_MAP[
                    document.metadata["file_type"]
                ]
                yield document
            except Exception as e:
                logger.warning(f"Failed to load code in {f_name} with error {e}")


def load(
    project: str,
    entity: str,
    result_artifact_name: str = "raw_dataset",
):
    run = wandb.init(project=project, entity=entity, job_type="prepare_dataset")
    artifact = wandb.Artifact(
        result_artifact_name, type="dataset", description="Raw documents for wandbot"
    )

    en_docodile_loader = DocodileDataLoader(DocodileEnglishStoreConfig())
    ja_docodile_loader = DocodileDataLoader(DocodileJapaneseStoreConfig())
    examples_code_loader = CodeDataLoader(ExampleCodeStoreConfig())
    examples_notebook_loader = CodeDataLoader(ExampleNotebookStoreConfig())
    sdk_code_loader = CodeDataLoader(SDKCodeStoreConfig())
    # weave_code_loader = CodeDataLoader(WeaveCodeStoreConfig())

    for loader in [
        en_docodile_loader,
        ja_docodile_loader,
        examples_code_loader,
        examples_notebook_loader,
        sdk_code_loader,
        # weave_code_loader,
    ]:
        loader.config.docstore_dir.mkdir(parents=True, exist_ok=True)
        with (loader.config.docstore_dir / "metadata.json").open("w") as f:
            json.dump(loader.metadata, f)

        with (loader.config.docstore_dir / "config.json").open("w") as f:
            f.write(loader.config.model_dump_json())

        with (loader.config.docstore_dir / "documents.jsonl").open("w") as f:
            for document in loader.load():
                document_json = {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                f.write(json.dumps(document_json) + "\n")

        artifact.add_dir(
            str(loader.config.docstore_dir), name=loader.config.docstore_dir.name
        )
    run.log_artifact(artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"


def main():
    load(
        project=os.environ.get("WANDB_PROJECT", "wandbot-dev"),
        entity=os.environ.get("WANDB_ENTITY", "wandbot"),
    )


if __name__ == "__main__":
    main()
