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
        )
    except Exception as e:
        logger.error(f"Failed to fetch metadata from repo at {repo.working_dir}: {e}")
        return {
            "commit_summary": "Error",
            "commit_message": "Error fetching metadata",
            "commit_author": "N/A",
            "commit_time": "N/A",
            "commit_hash": "N/A",
        }


def fetch_git_repo(paths: DataSource, id_file: Path) -> Dict[str, str]:
    """Fetch the git repository using direct subprocess calls for clone and LFS.

    Clones the repository fully for the specified branch after ensuring
    any existing local directory is removed. Includes Git LFS pull steps.

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
    branch = paths.branch if paths.branch else "main"

    repo_metadata = {}

    try:
        # Ensure the parent directory exists
        local_repo_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Delete existing directory ---
        if local_repo_path.exists():
            logger.warning(f"Removing existing directory {local_repo_path} to perform a fresh clone.")
            try:
                shutil.rmtree(local_repo_path)
            except OSError as e:
                logger.error(f"Failed to remove existing directory {local_repo_path}: {e}")
                raise

        # --- Use subprocess for git clone ---
        logger.info(f"Performing fresh clone of {repo_url} (branch: {branch}) into {local_repo_path} using subprocess...")
        clone_cmd = [
            'git', 'clone', '--branch', branch,
            '--depth=1',
            '--', repo_url, str(local_repo_path)
        ]
        logger.info(f"Running clone command: {' '.join(clone_cmd)}")
        result = subprocess.run(clone_cmd, env=git_ssh_command_env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error(f"Subprocess git clone failed. Return code: {result.returncode}")
            logger.error(f"Clone stdout: {result.stdout}")
            logger.error(f"Clone stderr: {result.stderr}")
            raise GitCommandError(clone_cmd, result.returncode, result.stderr)
        logger.info(f"Successfully cloned branch '{branch}' of {repo_url} via subprocess.")
        logger.debug(f"Clone stdout: {result.stdout}")
        logger.debug(f"Clone stderr: {result.stderr}")

        # --- Use subprocess for Git LFS pull steps ---
        logger.info(f"Attempting Git LFS pull for {local_repo_path} using subprocess")
        try:
            # Need commit hash for fetch - use GitPython just for this temporarily, or parse from clone output if possible.
            # For now, let's try fetching all LFS objects for the branch tip.
            # repo = Repo(local_repo_path) # Re-initialize Repo object to get commit hash
            # commit_hash = repo.head.commit.hexsha

            # Fetch all LFS objects for the current checkout
            lfs_fetch_cmd = ['git', 'lfs', 'fetch']
            logger.info(f"Running LFS fetch command: {' '.join(lfs_fetch_cmd)}")
            result_fetch = subprocess.run(lfs_fetch_cmd, cwd=local_repo_path, env=git_ssh_command_env, capture_output=True, text=True, check=False)
            if result_fetch.returncode != 0:
                logger.warning(f"Subprocess git lfs fetch failed. Return code: {result_fetch.returncode}. Stderr: {result_fetch.stderr}. This might be ok.")
            else:
                 logger.info(f"LFS fetch stdout: {result_fetch.stdout}")
                 logger.debug(f"LFS fetch stderr: {result_fetch.stderr}")

            # Checkout LFS files (replace pointers)
            lfs_checkout_cmd = ['git', 'lfs', 'checkout']
            logger.info(f"Running LFS checkout command: {' '.join(lfs_checkout_cmd)}")
            result_checkout = subprocess.run(lfs_checkout_cmd, cwd=local_repo_path, env=git_ssh_command_env, capture_output=True, text=True, check=False)
            if result_checkout.returncode != 0:
                 logger.warning(f"Subprocess git lfs checkout failed. Return code: {result_checkout.returncode}. Stderr: {result_checkout.stderr}. This might be ok.")
            else:
                 logger.info(f"LFS checkout stdout: {result_checkout.stdout}")
                 logger.debug(f"LFS checkout stderr: {result_checkout.stderr}")

            logger.info(f"Git LFS pull commands executed via subprocess for {local_repo_path}")

        except Exception as e_lfs_sub:
            logger.warning(f"An unexpected error occurred during subprocess Git LFS operations: {e_lfs_sub}")
        # --- End Git LFS pull steps ---

        # --- Add Debugging: List directory contents ---
        try:
            logger.debug(f"DEBUG: Listing contents of cloned repo at {local_repo_path} (after subprocess clone/LFS):")
            for root, dirs, files in os.walk(local_repo_path):
                if '.git' in dirs:
                    dirs.remove('.git')
                relative_root = os.path.relpath(root, local_repo_path)
                if relative_root == '.':
                    relative_root = ''
                for name in files:
                    logger.debug(f"  Listing files: {os.path.join(relative_root, name)}")
                for name in dirs:
                    logger.debug(f"  Lising dirs:  {os.path.join(relative_root, name)}/")
        except Exception as e_walk:
            logger.error(f"DEBUG: Failed to list directory contents for {local_repo_path}: {e_walk}")
        # --- End Debugging ---

        # Fetch metadata - Use GitPython Repo object initialized after successful subprocess clone
        try:
            repo = Repo(local_repo_path) # Initialize repo object *after* successful clone
            repo_metadata = fetch_repo_metadata(repo)
        except InvalidGitRepositoryError:
             logger.error(f"Path {local_repo_path} is not a valid Git repository after subprocess clone.")
             repo_metadata = {"error": "Invalid git repo after subprocess clone"}
        except Exception as e_meta:
            logger.error(f"Failed to get metadata after subprocess clone: {e_meta}")
            repo_metadata = {"error": "Metadata fetch failed after subprocess clone"}

    except FileNotFoundError as e: # Outer except for SSH key
        logger.error(f"SSH Key file error: {e}")
        raise # Reraise critical error
    except GitCommandError as e: # Outer except for clone failure
         logger.error(f"Git clone command failed: {e}")
         raise
    except Exception as e: # Outer except for other errors
        logger.error(f"An unexpected error occurred during git operations for {repo_url} at {local_repo_path}: {e}")
        raise

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
