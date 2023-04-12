import hashlib
import logging
import pathlib
import subprocess
from typing import Dict, Union

import regex as re
from git import Repo
from giturlparse import parse
from langchain.vectorstores import Chroma
from pydantic import AnyHttpUrl

from src.wandbot.ingestion.settings import BaseDataConfig

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


def fetch_repo_metadata(repo: Repo) -> Dict[str, str]:
    head_commit = repo.head.commit

    return dict(
        commit_summary=head_commit.summary,
        commit_message=head_commit.message,
        commit_author=str(head_commit.author),
        commit_time=head_commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        commit_hash=head_commit.hexsha,
        commit_stats=head_commit.stats.total,
    )


def fetch_git_repo(paths: BaseDataConfig, id_file) -> Dict[str, str]:
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
                for chunk in iter(lambda: f.read(4096), b""):
                    computed_hash.update(chunk)
        elif path.is_dir():
            computed_hash = md5_update_from_dir(path, file_pattern, computed_hash)
    return computed_hash


def md5_dir(directory, file_pattern=None):
    return md5_update_from_dir(directory, file_pattern, hashlib.md5()).hexdigest()


class ChromaWithEmbeddings(Chroma):
    def add_texts_and_embeddings(self, documents, embeddings, ids, metadatas):
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
