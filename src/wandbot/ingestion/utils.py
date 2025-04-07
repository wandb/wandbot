"""This module contains utility functions for the Wandbot ingestion system.

The module includes the following functions:
- `convert_contents_to_soup`: Converts contents to a BeautifulSoup object.
- `clean_soup`: Cleans the BeautifulSoup object.
- `clean_contents`: Cleans the contents.
- `get_git_command`: Get the git command with the given id file.
- `fetch_git_remote_hash`: Fetches the remote hash of the git repository.
- `fetch_repo_metadata`: Fetches the metadata of the git repository.
- `fetch_git_repo`: Fetches the git repository.
- `concatenate_cells`: Combines cells information in a readable format.

The module also includes the following constants:
- `EXTENSION_MAP`: A dictionary mapping file extensions to programming languages.

Typical usage example:

    contents = "This is some markdown content."
    soup = convert_contents_to_soup(contents)
    cleaned_soup = clean_soup(soup)
    cleaned_contents = clean_contents(contents)
    git_command = get_git_command(id_file)
    remote_hash = fetch_git_remote_hash(repo_url, id_file)
    repo_metadata = fetch_repo_metadata(repo)
    git_repo_metadata = fetch_git_repo(paths, id_file)
    cell_info = concatenate_cells(cell, include_outputs, max_output_length, traceback)
"""

import pathlib
import re
import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import filelock

import frontmatter
import giturlparse
import markdown
import markdownify
from bs4 import BeautifulSoup, Comment
from git import Repo, InvalidGitRepositoryError, GitCommandError

from wandbot.utils import get_logger
from wandbot.configs.ingestion_config import DataSource

logger = get_logger(__name__)


def get_git_command(id_file: Path) -> str:
    """Get the git command with the given id file.

    Args:
        id_file: The path to the id file.

    Returns:
        The git command with the id file.
    """
    if not id_file.is_file():
        raise FileNotFoundError(f"SSH key file not found: {id_file}")

    git_command = f"ssh -v -i {id_file.resolve()}"  # Use resolve() for absolute path
    return git_command


def fetch_git_remote_hash(
    repo_url: Optional[str] = None, id_file: Optional[Path] = None
) -> Optional[str]:
    """Fetch the remote hash of the git repository.

    Args:
        repo_url: The URL of the git repository.
        id_file: The path to the id file.

    Returns:
        The remote hash of the git repository.
    """
    if repo_url is None:
        logger.warning("No repo url was supplied. Not returning a repo hash")
        return None
    git_command = get_git_command(id_file)
    repo_url = giturlparse.parse(repo_url).urls.get("ssh")

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
    """Fetch the metadata of the git repository.

    Args:
        repo: The git repository.

    Returns:
        The metadata of the git repository.
    """
    try:
        head_commit = repo.head.commit
        return dict(
            commit_summary=head_commit.summary,
            commit_message=head_commit.message,
            commit_author=str(head_commit.author),
            commit_time=head_commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            commit_hash=head_commit.hexsha,
            commit_stats=head_commit.stats.total,
        )
    except Exception as e:
        logger.error(f"Failed to fetch metadata from repo at {repo.working_dir}: {e}")
        return {
            "commit_summary": "Error",
            "commit_message": "Error fetching metadata",
            "commit_author": "N/A",
            "commit_time": "N/A",
            "commit_hash": "N/A",
            "commit_stats": {},
        }


def fetch_git_repo(paths: DataSource, id_file: Path) -> Dict[str, str]:
    """Fetch the git repository, optimizing for single clone per repo URL.

    Clones the repository if it doesn't exist locally, otherwise fetches updates
    and checks out the specified branch. Assumes configurations for the same
    repo_path use the same local_path.
    Uses file locking to prevent race conditions during concurrent access.

    Args:
        paths: The DataSource configuration containing repo_path, local_path, branch, etc.
        id_file: The path to the id file (SSH key).

    Returns:
        The metadata of the git repository after checkout.
    """
    git_command = get_git_command(id_file)
    git_ssh_command_env = os.environ.copy()
    git_ssh_command_env["GIT_SSH_COMMAND"] = git_command

    local_repo_path = paths.local_path
    repo_url = paths.repo_path
    # Use 'main' as default if branch is not specified or None
    branch = paths.branch if paths.branch else "main"

    # Define a lock file path based on the local repo path
    lock_path = local_repo_path.parent / f"{local_repo_path.name}.lock"

    # Ensure the parent directory for the clone exists
    local_repo_path.parent.mkdir(parents=True, exist_ok=True)

    repo = None
    repo_metadata = {}

    try:
        # Acquire lock for this specific repository path
        with filelock.FileLock(str(lock_path)):
            logger.info(f"Acquired lock for {local_repo_path}")

            if local_repo_path.is_dir():
                try:
                    repo = Repo(local_repo_path)
                    logger.info(
                        f"Repo {local_repo_path} already exists. Fetching updates for {repo_url}"
                    )
                    with repo.git.custom_environment(GIT_SSH_COMMAND=git_command):
                        origin = repo.remotes.origin
                        # Check and configure fetch refspec if missing
                        if not origin.config_reader.has_option('fetch'):
                            logger.warning(f"Missing fetch refspec for remote 'origin' in {local_repo_path}. Configuring default.")
                            with origin.config_writer as writer:
                                writer.set("fetch", "+refs/heads/*:refs/remotes/origin/*")

                        # Fetch all remote branches refs without checking out/merging
                        logger.info(f"Fetching updates for {local_repo_path}...")
                        origin.fetch(prune=True)
                        logger.info(f"Fetch complete for {local_repo_path}.")

                except InvalidGitRepositoryError:
                    logger.warning(
                        f"Folder {local_repo_path} exists but is not a valid git repo. Removing it."
                    )
                    try:
                        shutil.rmtree(local_repo_path)
                    except OSError as e:
                        logger.error(f"Failed to remove existing directory {local_repo_path}: {e}")
                        raise
                    repo = None # Force clone below

            if repo is None:
                logger.info(f"Cloning {repo_url} (branch: {branch}) into {local_repo_path}...")
                # Initial clone - try cloning the specific branch directly first for efficiency
                try:
                     repo = Repo.clone_from(
                         url=repo_url,
                         to_path=local_repo_path,
                         env=git_ssh_command_env,
                         branch=branch,
                         # depth=1 # Optional: Uncomment for shallow clone if full history isn't needed
                     )
                     logger.info(f"Successfully cloned branch '{branch}' of {repo_url}")
                except GitCommandError as e:
                    # If cloning a specific branch fails (e.g., during first setup or transient error)
                    # Fallback: Clone default branch first, then checkout desired branch
                    logger.warning(f"Failed to clone branch '{branch}' directly for {repo_url}. "
                                   f"Cloning default branch and then checking out '{branch}'. Error: {e}")
                    # Check if clone dir exists from failed attempt and remove if necessary
                    if local_repo_path.exists():
                        logger.warning(f"Removing potentially incomplete directory {local_repo_path} before retrying.")
                        shutil.rmtree(local_repo_path)

                    repo = Repo.clone_from(
                        url=repo_url,
                        to_path=local_repo_path,
                        env=git_ssh_command_env,
                        # Don't specify branch here, clones default (usually 'main' or 'master')
                        # depth=1 # Optional: Uncomment for shallow clone
                    )
                    logger.info(f"Successfully cloned default branch of {repo_url}. Will checkout '{branch}' next.")
                    # We'll checkout the desired branch in the next step

                # Now checkout and pull the desired branch
                with repo.git.custom_environment(GIT_SSH_COMMAND=git_command):
                    logger.info(f"Checking out branch: {branch} in {local_repo_path}")
                    try:
                        repo.git.checkout(branch)
                        logger.info(f"Successfully checked out branch: {branch}")
                    except GitCommandError as e:
                        # Try fetching again and then checkout, branch might be new
                        logger.warning(f"Initial checkout of branch '{branch}' failed: {e}. Fetching again and retrying checkout.")
                        try:
                            repo.remotes.origin.fetch()
                            repo.git.checkout(branch)
                            logger.info(f"Successfully checked out branch: {branch} after second attempt.")
                        except GitCommandError as e_retry:
                             logger.error(f"Failed to checkout branch '{branch}' even after fetching again in {local_repo_path}: {e_retry}. "
                                          f"Proceeding with current branch '{repo.active_branch.name}'.")
                             branch = repo.active_branch.name # Update branch variable to reflect reality

                    # Pull the latest changes for the target branch (either original or fallback)
                    logger.info(f"Pulling latest changes for branch: {branch} in {local_repo_path}")
                    try:
                        repo.remotes.origin.pull(branch)
                        logger.info(f"Successfully pulled changes for branch: {branch}")
                    except GitCommandError as e:
                         logger.error(f"Failed to pull changes for branch '{branch}' in {local_repo_path}: {e}")
                         # Proceed with current state, metadata will reflect this

                # Fetch metadata *inside* the lock
                repo_metadata = fetch_repo_metadata(repo)
                logger.info(f"Releasing lock for {local_repo_path}")

    except FileNotFoundError as e:
        logger.error(f"SSH Key file error: {e}")
        raise # Reraise critical error
    except filelock.Timeout:
        logger.error(f"Could not acquire lock for {local_repo_path} within timeout period.")
        # Return empty metadata or raise an error, depending on desired behavior
        raise TimeoutError(f"Failed to acquire lock for git repo: {local_repo_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during git operations for {repo_url} at {local_repo_path}: {e}")
        raise # Reraise unexpected errors

    return repo_metadata


def concatenate_cells(
    cell: Dict[str, Any],
    include_outputs: bool,
    max_output_length: int,
    traceback: bool,
) -> str:
    """Combine cells information in a readable format ready to be used.

    Args:
        cell: The cell dictionary.
        include_outputs: Whether to include outputs.
        max_output_length: The maximum length of the output.
        traceback: Whether to include traceback.

    Returns:
        The combined cell information.
    """
    cell_type = cell["cell_type"]
    source = cell["source"]
    output = cell["outputs"]

    if include_outputs and cell_type == "code" and output:
        if "ename" in output[0].keys():
            error_name = output[0]["ename"]
            error_value = output[0]["evalue"]
            if traceback:
                traceback = output[0]["traceback"]
                return (
                    f"'{cell_type}' cell: '{source}'\n, gives error '{error_name}',"
                    f" with description '{error_value}'\n"
                    f"and traceback '{traceback}'\n\n"
                )
            else:
                return (
                    f"'{cell_type}' cell: '{source}'\n, gives error '{error_name}',"
                    f"with description '{error_value}'\n\n"
                )
        elif output[0]["output_type"] == "stream":
            output = output[0]["text"]
            min_output = min(max_output_length, len(output))
            return (
                f"'{cell_type}' cell: '{source}'\n with "
                f"output: '{output[:min_output]}'\n\n"
            )
    else:
        if cell_type == "markdown":
            source = re.sub(r"!\[.*?\]\((.*?)\)", "", f"{source}").strip()
            if source and len(source) > 5:
                return f"'{cell_type}' cell: '{source}'\n\n"
        else:
            return f"'{cell_type}' cell: '{source}'\n\n"

    return ""


EXTENSION_MAP: Dict[str, str] = {
    ".py": "python",
    ".ipynb": "python",
    ".md": "markdown",
    ".js": "javascript",
    ".ts": "typescript",
}


def convert_contents_to_soup(contents: str) -> BeautifulSoup:
    """Converts contents to BeautifulSoup object.

    Args:
        contents: The contents to convert.

    Returns:
        The BeautifulSoup object.
    """
    markdown_document = markdown.markdown(
        contents,
        extensions=[
            "toc",
            "pymdownx.extra",
            "pymdownx.blocks.admonition",
            "pymdownx.magiclink",
            "pymdownx.blocks.tab",
            "pymdownx.pathconverter",
            "pymdownx.saneheaders",
            "pymdownx.striphtml",
        ],
    )
    soup = BeautifulSoup(markdown_document, "html.parser")
    return soup


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Cleans the BeautifulSoup object.

    Args:
        soup: The BeautifulSoup object to clean.

    Returns:
        The cleaned BeautifulSoup object.
    """
    for img_tag in soup.find_all("img", src=True):
        img_tag.extract()
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    for p_tag in soup.find_all("p"):
        if not p_tag.text.strip():
            p_tag.decompose()
    return soup


def clean_contents(contents: str) -> str:
    """Cleans the contents.

    Args:
        contents: The contents to clean.

    Returns:
        The cleaned contents.
    """
    soup = convert_contents_to_soup(contents)
    soup = clean_soup(soup)
    cleaned_document = markdownify.MarkdownConverter(
        heading_style="ATX"
    ).convert_soup(soup)
    # Regular expression pattern to match import lines
    js_import_pattern = r"import .* from [‘’']@theme/.*[‘’'];\s*\n*"
    cleaned_document = re.sub(js_import_pattern, "", cleaned_document)
    cleaned_document = cleaned_document.replace("![]()", "\n")
    cleaned_document = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cleaned_document)
    cleaned_document = re.sub(r"\n{3,}", "\n\n", cleaned_document)
    try:
        cleaned_document = frontmatter.loads(cleaned_document).content
    except Exception as e:
        logger.warning(f"Could not parse or remove frontmatter: {e}. Proceeding with raw content including potential frontmatter.")
    return cleaned_document


def extract_frontmatter(file_path: pathlib.Path) -> Dict[str, Any]:
    """Extracts the frontmatter from a file.

    Args:
        file_path: The path to the file.

    Returns:
        The extracted frontmatter.
    """
    with open(file_path, "r") as f:
        contents = frontmatter.load(f)
        return {k: contents[k] for k in contents.keys()}
