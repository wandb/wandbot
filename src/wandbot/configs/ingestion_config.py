"""This module defines the configuration for data sources and stores in the Wandbot ingestion system.

This module contains classes that define the configuration for various data sources and stores used in the Wandbot
ingestion system. Each class represents a different type of data source or store, such as English and Japanese
documentation, example code, SDK code, and more. Each class is defined with various attributes like name,
data source, docstore directory, etc.

Typical usage example:

  data_store_config = DataStoreConfig()
  docodile_english_store_config = DocodileEnglishStoreConfig()
"""

import datetime
import pathlib
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from wandbot.utils import get_logger

logger = get_logger(__name__)


class IngestionConfig(BaseSettings):
    wandb_project: str = Field("wandbot-dev", env="WANDB_PROJECT")
    wandb_entity: str = Field("wandbot", env="WANDB_ENTITY")
    vectorstore_index_artifact_name: str = Field("chroma_index")
    vectorstore_index_artifact_type: str = Field("vectorstore")
    cache_dir: pathlib.Path = Field(
        pathlib.Path("data/cache/"), env="WANDBOT_CACHE_DIR"
    )


class DataSource(BaseSettings):
    cache_dir: pathlib.Path = Field("data/cache/raw_data")
    ignore_cache: bool = False
    remote_path: str = ""
    repo_path: str = ""
    local_path: Optional[pathlib.Path] = None
    branch: Optional[str] = None
    base_path: Optional[str] = ""
    file_patterns: List[str] = ["*.*"]
    is_git_repo: bool = False
    git_id_file: Optional[pathlib.Path] = Field(None, env="WANDBOT_GIT_ID_FILE")


class DataStoreConfig(BaseModel):
    name: str = "docstore"
    source_type: str = ""
    data_source: DataSource = DataSource()
    docstore_dir: pathlib.Path = pathlib.Path("docstore")
    chunk_size: int = 512
    chunk_multiplier: int = 2
    language: Optional[str] = None

    @model_validator(mode="after")
    def _set_cache_paths(cls, values: "DataStoreConfig") -> "DataStoreConfig":
        sanitized_name = values.name.replace(" ", "_")
        values.docstore_dir = (
            values.data_source.cache_dir / sanitized_name / values.docstore_dir
        )
        data_source = values.data_source

        if data_source.repo_path:
            data_source.is_git_repo = (
                urlparse(data_source.repo_path).netloc == "github.com"
            )
            local_path_name = urlparse(data_source.repo_path).path.split("/")[-1]
            if not data_source.local_path:
                data_source.local_path = (
                    data_source.cache_dir / sanitized_name / local_path_name
                )
            if data_source.is_git_repo:
                if data_source.git_id_file is None:
                    logger.debug(
                        "The source data is a git repo but no git_id_file is set."
                        " Attempting to use the default ssh id file"
                    )
                    data_source.git_id_file = pathlib.Path.home() / ".ssh" / "id_rsa"
        values.data_source = data_source

        return values


class DocodileEnglishStoreConfig(DataStoreConfig):
    name: str = "english_documentation"
    source_type: str = "documentation"
    data_source: DataSource = DataSource(
        remote_path="https://docs.wandb.ai/",
        repo_path="https://github.com/wandb/docs",
        branch="main",
        base_path="content",
        file_patterns=["*.md", "*.mdx"],
        is_git_repo=True,
    )
    language: str = "en"
    docstore_dir: pathlib.Path = pathlib.Path("wandb_documentation_en")
    chunk_size: int = 768 // 2
    chunk_multiplier: int = 2


class DocodileJapaneseStoreConfig(DataStoreConfig):
    name: str = "japanese_documentation"
    source_type: str = "documentation"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/docs/",
        repo_path="https://github.com/wandb/docs/",
        base_path="docs",
        file_patterns=["*.md", "*.mdx"],
        is_git_repo=True,
        branch="japanese_docs",
    )
    language: str = "ja"
    docstore_dir: pathlib.Path = pathlib.Path("wandb_documentation_ja")


class DocodileKoreanStoreConfig(DataStoreConfig):
    name: str = "korean_documentation"
    source_type: str = "documentation"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/docs/",
        repo_path="https://github.com/wandb/docs/",
        base_path="docs",
        file_patterns=["*.md"],
        is_git_repo=True,
        branch="korean_docs",
    )
    language: str = "ko"
    docstore_dir: pathlib.Path = pathlib.Path("wandb_documentation_ko")


class SDKCodeStoreConfig(DataStoreConfig):
    name: str = "wandb_sdk_code"
    source_type: str = "code"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/wandb/tree/main/",
        repo_path="https://github.com/wandb/wandb",
        branch="main",
        base_path="wandb",
        file_patterns=["*.py"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("wandb_sdk_code")


class SDKTestsStoreConfig(DataStoreConfig):
    name: str = "wandb_sdk_tests"
    source_type: str = "code"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/wandb/tree/main/",
        repo_path="https://github.com/wandb/wandb",
        branch="main",
        base_path="tests",
        file_patterns=["*.py"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("wandb_sdk_tests")


class WeaveDocStoreConfig(DataStoreConfig):
    name: str = "weave_documentation"
    source_type: str = "documentation"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/weave",
        repo_path="https://github.com/wandb/weave",
        branch="master",
        base_path="docs",
        file_patterns=["*.md", "*.mdx"],
        is_git_repo=True,
    )
    language: str = "en"
    docstore_dir: pathlib.Path = pathlib.Path("weave_documentation")


class WeaveCookbookStoreConfig(DataStoreConfig):
    name: str = "weave_cookbooks"
    source_type: str = "code"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/weave",
        repo_path="https://github.com/wandb/weave",
        branch="master",
        base_path="docs/notebooks",
        file_patterns=["*.ipynb"],
        is_git_repo=True,
    )
    language: str = "en"
    docstore_dir: pathlib.Path = pathlib.Path("weave_cookbooks")


class WeaveCodeStoreConfig(DataStoreConfig):
    name: str = "weave_sdk_code"
    source_type: str = "code"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/weave/tree/master/",
        repo_path="https://github.com/wandb/weave",
        branch="master",
        base_path="weave",
        file_patterns=["*.py", "*.ipynb"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("weave_sdk_code")


class ExampleCodeStoreConfig(DataStoreConfig):
    name: str = "examples_code"
    source_type: str = "code"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/examples/tree/master/",
        repo_path="https://github.com/wandb/examples",
        branch="master",
        base_path="examples",
        file_patterns=["*.py"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("wandb_examples_code")


class ExampleNotebookStoreConfig(DataStoreConfig):
    name: str = "examples_notebooks"
    source_type: str = "code"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/examples/tree/master/",
        repo_path="https://github.com/wandb/examples",
        branch="master",
        base_path="colabs",
        file_patterns=["*.ipynb"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("wandb_examples_colab")


class WandbEduCodeStoreConfig(DataStoreConfig):
    name: str = "wandb_edu_code"
    source_type: str = "code"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/edu/tree/main/",
        repo_path="https://github.com/wandb/edu",
        branch="main",
        base_path="",
        file_patterns=["*.py", "*.ipynb"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("wandb_edu_code")


class WandbEduDocStoreConfig(DataStoreConfig):
    name: str = "wandb_edu_text"
    source_type: str = "documentation"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/edu/tree/main/",
        repo_path="https://github.com/wandb/edu",
        branch="main",
        base_path="",
        file_patterns=["*.md"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("wandb_edu_text")


class WeaveJsStoreConfig(DataStoreConfig):
    name: str = "weave_js_code"
    source_type: str = "code"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/weave/tree/master/",
        repo_path="https://github.com/wandb/weave",
        branch="master",
        base_path="weave-js",
        file_patterns=["*.js", "*.ts"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("weave_js")


class FCReportsStoreConfig(DataStoreConfig):
    name: str = "fc_reports"
    source_type: str = "report"
    data_source: DataSource = DataSource(
        remote_path="wandb-production",
        repo_path="",
        base_path="reports",
        file_patterns=["*.json"],
        is_git_repo=False,
    )
    docstore_dir: pathlib.Path = pathlib.Path("fc_reports")

    @model_validator(mode="after")
    def _set_cache_paths(cls, values: "DataStoreConfig") -> "DataStoreConfig":
        sanitized_name = values.name.replace(" ", "_")
        values.docstore_dir = (
            values.data_source.cache_dir / sanitized_name / values.docstore_dir
        )
        data_source = values.data_source

        data_source.local_path = (
            data_source.cache_dir
            / sanitized_name
            / f"reports_{int(datetime.datetime.now().timestamp())}.json"
        )
        values.data_source = data_source

        return values