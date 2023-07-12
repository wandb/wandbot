import datetime
import hashlib
import logging
import pathlib
import subprocess
import zipfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import regex as re
import wandb
from giturlparse import parse
from langchain.schema import Document
from llama_index import Document as LlamaDocument
from pydantic import AnyHttpUrl

if TYPE_CHECKING:
    from git import Repo
    from wandbot.ingestion.datastore import DataStore

logger = logging.getLogger(__name__)


def get_git_command(id_file):
    assert id_file.is_file()

    git_command = f"ssh -v -i /{id_file}"
    return git_command


def fetch_git_remote_hash(repo_url=None, id_file=None):
    if repo_url is None:
        logger.warning(f"No repo url was supplied. Not returning a repo hash")
        return None
    git_command = get_git_command(id_file)
    repo_url = parse(repo_url).urls.get("ssh")

    cmd = f'GIT_SSH_COMMAND="{git_command} -o IdentitiesOnly=yes" git ls-remote {repo_url}'
    normal = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    sha = re.split(r"\t+", normal.stdout)[0]
    return sha


def fetch_repo_metadata(repo: "Repo") -> Dict[str, str]:
    head_commit = repo.head.commit

    return dict(
        commit_summary=head_commit.summary,
        commit_message=head_commit.message,
        commit_author=str(head_commit.author),
        commit_time=head_commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        commit_hash=head_commit.hexsha,
        commit_stats=head_commit.stats.total,
    )


def fetch_git_repo(paths, id_file) -> Dict[str, str]:
    from git import Repo

    git_command = get_git_command(id_file)

    if paths.local_path.is_dir():
        repo = Repo(paths.local_path)
        logger.debug(
            f"Repo {paths.local_path} already exists... Pulling changes from {repo.remotes.origin.url}"
        )
        with repo.git.custom_environment(GIT_SSH_COMMAND=git_command):
            repo.remotes.origin.pull()
    else:
        remote_url = parse(paths.repo_path).urls.get("ssh")

        logger.debug(f"Cloning {remote_url} to {paths.local_path}")
        repo = Repo.clone_from(
            remote_url, paths.local_path, env=dict(GIT_SSH_COMMAND=git_command)
        )
    return fetch_repo_metadata(repo)


def map_local_to_remote(
    paths, base_name: str, remote_path: Union[str, AnyHttpUrl]
) -> Dict[str, str]:
    paths = list(paths)
    local_paths = list(map(lambda x: str(x), paths))
    local_path_parts = list(map(lambda x: x.parts, paths))
    examples_idx = list(map(lambda x: x.index(base_name), local_path_parts))
    remote_paths = list(
        map(
            lambda x: "/".join(x[1][x[0] + 1 :]),
            zip(examples_idx, local_path_parts),
        )
    )
    remote_paths = list(
        map(
            lambda x: f"{remote_path}{x}",
            remote_paths,
        )
    )
    return dict(zip(local_paths, remote_paths))


def md5_update_from_dir(directory, file_pattern, computed_hash):
    # ref: https://stackoverflow.com/a/54477583
    assert pathlib.Path(directory).is_dir()
    if file_pattern is not None:
        path_iterator = pathlib.Path(directory).glob(file_pattern)
    else:
        path_iterator = pathlib.Path(directory).iterdir()
    for path in sorted(path_iterator, key=lambda p: str(p).lower()):
        computed_hash.update(path.name.encode())
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(), b""):
                    computed_hash.update(chunk)
        elif path.is_dir():
            computed_hash = md5_update_from_dir(path, file_pattern, computed_hash)
    return computed_hash


def md5_dir(directory, file_pattern=None):
    return md5_update_from_dir(directory, file_pattern, hashlib.md5()).hexdigest()


def add_metadata_to_documents(
    documents: List[Document], source_map: Optional[Dict[str, str]] = None
) -> List[Document]:
    out_documents = []
    for document in documents:
        if isinstance(source_map, dict):
            source = source_map.get(
                document.metadata["source"], document.metadata["source"]
            )
        else:
            source = document.metadata["source"]
        doc_id = hashlib.md5(
            (source + document.page_content).encode("UTF-8")
        ).hexdigest()
        metadata = {"source": source, "doc_id": doc_id}
        out_documents.append(
            Document(page_content=document.page_content, metadata=metadata)
        )
    return out_documents


def convert_llama_docstore_to_vectorstore_kwargs(
    documents: Dict[str, LlamaDocument]
) -> Dict[str, Any]:
    docs_dict = {}
    for doc_id, document in documents.items():
        docs_dict["ids"] = docs_dict.get("ids", []) + [doc_id]
        docs_dict["documents"] = docs_dict.get("documents", []) + [
            document.to_langchain_format()
        ]
    return docs_dict


def load_docstore_class(module, cls: str):
    return getattr(module, cls)


class Timer:
    def __init__(self) -> None:
        self.start = datetime.datetime.utcnow()
        self.stop = self.start

    def __enter__(self) -> "Timer":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop = datetime.datetime.utcnow()

    @property
    def elapsed(self) -> float:
        return (self.stop - self.start).total_seconds()


def save_dataset(data_sources: List["DataStore"]):
    artifact = wandb.Artifact("raw_dataset", type="dataset")
    for data_source in data_sources:
        archive_name = (
            f"{data_source.config.data_source.cache_dir}/{data_source.config.name}.zip"
        )
        with zipfile.ZipFile(archive_name, "w") as f:
            for file in data_source.config.data_source.local_path.rglob("**/*"):
                if not file.is_symlink():
                    f.write(file)
        artifact.add_file(archive_name)

    wandb.log_artifact(
        artifact,
    )
    return artifact.wait()
