import pathlib
from typing import Optional, Union

from pydantic import (
    BaseSettings,
    Field,
    AnyHttpUrl,
    root_validator,
    BaseModel,
)


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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DocStoreConfig(BaseModel):
    name: str = "docstore"
    data_config: BaseDataConfig = BaseDataConfig()
    chunk_size: int = 1024
    chunk_overlap: int = 0
    encoding_name: str = "cl100k_base"
    hyde_temperature: bool = 0.5
    hyde_prompt: pathlib.Path = pathlib.Path("../data/prompts/hyde_prompt.txt")
    docstore_file: str = "docstore.json"
    vectorstore_dir: str = "vectorstore"
    wandb_project: str = "wandb_docs_bot"


class DocumentationStoreConfig(DocStoreConfig):
    name: str = "documentation_store"
    data_config: BaseDataConfig = BaseDataConfig(
        remote_path="https://docs.wandb.ai/",
        repo_path="https://github.com/wandb/docodile",
        base_path="",
        file_pattern="*.md",
        is_git_repo=True,
    )

    @root_validator(pre=False)
    def _set_cache_paths(cls, values: dict) -> dict:
        values["docstore_file"] = (
            values["data_config"].cache_dir
            / f"{values['name']}/{values['docstore_file']}"
        )
        values["vectorstore_dir"] = (
            values["data_config"].cache_dir
            / f"{values['name']}/{values['vectorstore_dir']}"
        )
        data_config = values["data_config"].dict()

        if data_config["repo_path"] and isinstance(
            data_config["repo_path"], AnyHttpUrl
        ):
            data_config["is_git_repo"] = data_config["repo_path"].host == "github.com"
            local_path = data_config["repo_path"].path.split("/")[-1]
            data_config["local_path"] = (
                data_config["cache_dir"] / values["name"] / local_path
            )
        values["data_config"] = BaseDataConfig(**data_config)

        return values
