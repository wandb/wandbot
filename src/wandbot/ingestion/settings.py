import logging
import pathlib
from typing import Optional, Union, Dict

from pydantic import (
    BaseSettings,
    Field,
    AnyHttpUrl,
    root_validator,
    BaseModel,
)

logger = logging.getLogger(__name__)


class BaseDataConfig(BaseSettings):
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


class DocStoreConfig(BaseModel):
    name: str = "docstore"
    type: str = "document_store"
    cls: str = "BaseDocStore"
    data_config: BaseDataConfig = BaseDataConfig()
    chunk_size: int = 1024
    chunk_overlap: int = 0
    encoding_name: str = "cl100k_base"
    hyde_temperature: bool = 0.5
    hyde_prompt: pathlib.Path = pathlib.Path(
        "../data/prompts/hyde_prompt.txt"
    ).resolve()
    docstore_file: pathlib.Path = pathlib.Path("docstore.json")
    vectorstore_dir: pathlib.Path = pathlib.Path("vectorstore")
    wandb_project: str = "wandb_docs_bot"
    wandb_entity: Optional[str] = None

    @root_validator(pre=False)
    def _set_cache_paths(cls, values: dict) -> dict:
        values["docstore_file"] = (
            values["data_config"].cache_dir / values["name"] / values["docstore_file"]
        )
        values["vectorstore_dir"] = (
            values["data_config"].cache_dir / values["name"] / values["vectorstore_dir"]
        )
        data_config = values["data_config"].dict()

        if data_config["repo_path"] and isinstance(
            data_config["repo_path"], AnyHttpUrl
        ):
            data_config["is_git_repo"] = data_config["repo_path"].host == "github.com"
            local_path = data_config["repo_path"].path.split("/")[-1]
            if not data_config["local_path"]:
                data_config["local_path"] = (
                    data_config["cache_dir"] / values["name"] / local_path
                )

            if data_config["is_git_repo"]:
                if data_config["git_id_file"] is None:
                    logger.warning(
                        "The source data is a git repo but no git_id_file is set."
                        " Attempting to use the default ssh id file"
                    )
                    data_config["git_id_file"] = pathlib.Path.home() / ".ssh" / "id_rsa"
        values["data_config"] = BaseDataConfig(**data_config)

        return values


class DocumentationStoreConfig(DocStoreConfig):
    name: str = "documentation_store"
    cls: str = "DocumentationDocStore"
    data_config: BaseDataConfig = BaseDataConfig(
        remote_path="https://docs.wandb.ai/",
        repo_path="https://github.com/wandb/docodile",
        base_path="",
        file_pattern="*.md",
        is_git_repo=True,
    )


class CodeStoreConfig(DocStoreConfig):
    name: str = "code_store"
    cls: str = "CodeDocStore"
    data_config: BaseDataConfig = BaseDataConfig()


class ExamplesCodeStoreConfig(CodeStoreConfig):
    name: str = "examples_code_store"
    cls: str = "ExamplesCodeDocStore"
    data_config: BaseDataConfig = BaseDataConfig(
        remote_path="https://github.com/wandb/examples/blob/master/",
        repo_path="https://github.com/wandb/examples",
        base_path="examples",
        file_pattern="*.py",
        is_git_repo=True,
    )


class ExamplesNotebookStoreConfig(CodeStoreConfig):
    name: str = "examples_notebook_store"
    cls: str = "ExamplesNotebookDocStore"
    data_config: BaseDataConfig = BaseDataConfig(
        remote_path="https://github.com/wandb/examples/blob/master/",
        repo_path="https://github.com/wandb/examples",
        base_path="colabs",
        file_pattern="*.ipynb",
        is_git_repo=True,
    )


class SDKCodeStoreConfig(CodeStoreConfig):
    name: str = "sdk_code_store"
    cls: str = "SDKCodeDocStore"
    data_config: BaseDataConfig = BaseDataConfig(
        remote_path="https://github.com/wandb/wandb/blob/main/",
        repo_path="https://github.com/wandb/wandb",
        base_path="wandb",
        file_pattern="*.py",
        is_git_repo=True,
    )


class CsvStoreConfig(DocStoreConfig):
    name: str = "csv_store"
    cls: str = "CsvDocStore"
    data_config: BaseDataConfig = BaseDataConfig(
        file_pattern="*.csv",
        is_git_repo=False,
    )


class JsonlStoreConfig(DocStoreConfig):
    name: str = "jsonl_store"
    cls: str = "JsonlDocStore"
    data_config: BaseDataConfig = BaseDataConfig(
        file_pattern="*.jsonl",
        is_git_repo=False,
    )


class ExtraDataStoreConfig(JsonlStoreConfig):
    name: str = "extra_data_store"
    cls: str = "ExtraDataDocStore"
    data_config: BaseDataConfig = BaseDataConfig(
        local_path=pathlib.Path("../data/raw_dataset/extra_data").resolve(),
        file_pattern="*.jsonl",
        is_git_repo=False,
    )


class CombinedDocStoreConfig(DocStoreConfig):
    name: str = "combined_doc_store"
    type: str = "document_store"
    cls: str = "BaseCombinedDocStore"
    data_config: BaseDataConfig = BaseDataConfig()
    docstore_configs: Dict[str, DocStoreConfig] = {
        config().name: config()
        for config in [
            DocumentationStoreConfig,
            ExamplesCodeStoreConfig,
            ExamplesNotebookStoreConfig,
            SDKCodeStoreConfig,
            ExtraDataStoreConfig,
        ]
    }
    docstore_file: pathlib.Path = pathlib.Path("docstore.json")
    vectorstore_dir: pathlib.Path = pathlib.Path("vectorstore")
    wandb_project: str = "wandb_docs_bot"
    wandb_entity: Optional[str] = None

    @root_validator()
    def _set_cache_paths(cls, values: dict) -> dict:
        values["docstore_file"] = (
            values["data_config"].cache_dir / values["name"] / values["docstore_file"]
        )
        values["vectorstore_dir"] = (
            values["data_config"].cache_dir / values["name"] / values["vectorstore_dir"]
        )
        return values


class WandbotDocStoreConfig(CombinedDocStoreConfig):
    name: str = "wandbot_doc_store"
    cls: str = "WandbotDocStore"
