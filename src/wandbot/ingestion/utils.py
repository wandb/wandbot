import logging
import pathlib
import subprocess
from typing import Dict, Union

import regex as re
from git import Repo
from giturlparse import parse
from pydantic import AnyHttpUrl

from src.wandbot.ingestion.settings import BaseDataConfig

logger = logging.getLogger(__name__)


def get_git_command(id_file=None):
    if id_file is None:
        logger.warning(f"Using default ssh file")
        id_file = pathlib.Path.home() / ".ssh/id_rsa.pub"

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


def fetch_git_repo(paths: BaseDataConfig, id_file=None) -> Dict[str, str]:
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
