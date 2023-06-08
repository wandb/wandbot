import logging
import pathlib
from typing import Any, Dict, Optional, Union

from pydantic import AnyHttpUrl, BaseModel, BaseSettings, Field, root_validator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.WARNING
)
logger = logging.getLogger(__name__)


class DataSource(BaseSettings):
    cache_dir: pathlib.Path = Field("data/cache/", env="WANDBOT_CACHE_DIR")
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


class WeaveCodeStoreConfig(DataStoreConfig):
    name: str = "weave_code_store"
    data_source: DataSource = DataSource(
        remote_path="https://github.com/wandb/weave/blob/main/",
        repo_path="https://github.com/wandb/weave",
        file_pattern="*.*",
        is_git_repo=True,
    )


class ExtraDataStoreConfig(DataStoreConfig):
    name: str = "extra_data_store"
    data_source: DataSource = DataSource(
        local_path=pathlib.Path("data/raw_dataset/extra_data"),
        file_pattern="*.jsonl",
        is_git_repo=False,
    )


class GTMDataStoreConfig(DataStoreConfig):
    name: str = "gtm_data_store"
    data_source: DataSource = DataSource(
        local_path=pathlib.Path("data/raw_dataset/gtm_data"),
        file_pattern="*.*x",
        is_git_repo=False,
    )


class VectorIndexConfig(BaseSettings):
    name: str = "wandbot_vectorindex"
    cache_dir: pathlib.Path = Field(
        pathlib.Path("data/cache/"), env="WANDBOT_CACHE_DIR"
    )
    vectorindex_dir: pathlib.Path = Field("vectorindex", env="WANDBOT_VECTORINDEX_DIR")
    sparse_vectorizer_kwargs: Dict[str, Any] = {
        "max_df": 0.9,
        "ngram_range": (1, 3),
    }
    retrieval_size: int = 8
    wandb_project: str | None = Field(..., env="WANDBOT_WANDB_PROJECT")
    wandb_entity: str | None = Field(None, env="WANDBOT_WANDB_ENTITY")

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
