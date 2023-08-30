import pathlib
from typing import List, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
from wandbot.utils import get_logger

logger = get_logger(__name__)


class DataSource(BaseSettings):
    cache_dir: pathlib.Path = Field("data/cache/raw_data", env="WANDBOT_CACHE_DIR")
    ignore_cache: bool = False
    remote_path: str = ""
    repo_path: str = ""
    local_path: Optional[pathlib.Path] = None
    base_path: Optional[str] = ""
    file_pattern: Union[str, List[str]] = "*.*"
    is_git_repo: bool = False
    git_id_file: Optional[pathlib.Path] = Field(None, env="WANDBOT_GIT_ID_FILE")


class DataStoreConfig(BaseModel):
    name: str = "docstore"
    data_source: DataSource = DataSource()
    docstore_dir: pathlib.Path = pathlib.Path("docstore")

    @model_validator(mode="after")
    def _set_cache_paths(cls, values: "DataStoreConfig") -> "DataStoreConfig":
        values.docstore_dir = (
            values.data_source.cache_dir / values.name / values.docstore_dir
        )
        data_source = values.data_source

        if data_source.repo_path:
            data_source.is_git_repo = (
                urlparse(data_source.repo_path).netloc == "github.com"
            )
            local_path = urlparse(data_source.repo_path).path.split("/")[-1]
            if not data_source.local_path:
                data_source.local_path = (
                    data_source.cache_dir / values.name / local_path
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
    name: str = "docodile_store"
    data_source: DataSource = DataSource(
        remote_path="https://docs.wandb.ai/",
        repo_path="https://github.com/wandb/docodile",
        base_path="docs",
        file_pattern="*.md",
        is_git_repo=True,
    )
    language: str = "en"
    docstore_dir: pathlib.Path = pathlib.Path("docstore_en")


class DocodileJapaneseStoreConfig(DataStoreConfig):
    name: str = "docodile_store"
    data_source: DataSource = DataSource(
        remote_path="https://docs.wandb.ai/ja/",
        repo_path="https://github.com/wandb/docodile",
        base_path="i18n/ja/docusaurus-plugin-content-docs/current",
        file_pattern="*.md",
        is_git_repo=True,
    )
    language: str = "ja"
    docstore_dir: pathlib.Path = pathlib.Path("docstore_ja")


class ExampleCodeStoreConfig(DataStoreConfig):
    name: str = "examples_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/examples/tree/master/",
        repo_path="https://github.com/wandb/examples",
        base_path="examples",
        file_pattern="*.py",
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("docstore_example_code")


class ExampleNotebookStoreConfig(DataStoreConfig):
    name: str = "examples_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/examples/tree/master/",
        repo_path="https://github.com/wandb/examples",
        base_path="colabs",
        file_pattern="*.ipynb",
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("docstore_example_colab")


class SDKCodeStoreConfig(DataStoreConfig):
    name: str = "sdk_code_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/wandb/tree/main/",
        repo_path="https://github.com/wandb/wandb",
        base_path="wandb",
        file_pattern="*.py",
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("docstore_sdk_code")


class SDKTestsStoreConfig(DataStoreConfig):
    name: str = "sdk_code_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/wandb/tree/main/",
        repo_path="https://github.com/wandb/wandb",
        base_path="tests",
        file_pattern="*.py",
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("docstore_sdk_tests")


class WeaveCodeStoreConfig(DataStoreConfig):
    name: str = "weave_code_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/weave/tree/master/",
        repo_path="https://github.com/wandb/weave",
        base_path="weave",
        file_pattern=["*.py", "*.ipynb"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("docstore_weave_code")


class WeaveExamplesStoreConfig(DataStoreConfig):
    name: str = "weave_code_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/weave/tree/master/",
        repo_path="https://github.com/wandb/weave",
        base_path="examples",
        file_pattern=["*.py", "*.ipynb"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("docstore_weave_examples")


class WeaveJsStoreConfig(DataStoreConfig):
    name: str = "weave_code_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/weave/tree/master/",
        repo_path="https://github.com/wandb/weave",
        base_path="weave-js",
        file_pattern=["*.js", "*.ts"],
        is_git_repo=True,
    )
    docstore_dir: pathlib.Path = pathlib.Path("docstore_weave_js")
