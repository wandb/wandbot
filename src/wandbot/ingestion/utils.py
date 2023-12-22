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
from pathlib import Path
from typing import Any, Dict, Optional

import frontmatter
import giturlparse
import markdown
import markdownify
from bs4 import BeautifulSoup, Comment
from git import Repo
from wandbot.utils import get_logger

logger = get_logger(__name__)


def get_git_command(id_file: Path) -> str:
    """Get the git command with the given id file.

    Args:
        id_file: The path to the id file.

    Returns:
        The git command with the id file.
    """
    assert id_file.is_file()

    git_command = f"ssh -v -i /{id_file}"
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
        logger.warning(f"No repo url was supplied. Not returning a repo hash")
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
    head_commit = repo.head.commit

    return dict(
        commit_summary=head_commit.summary,
        commit_message=head_commit.message,
        commit_author=str(head_commit.author),
        commit_time=head_commit.committed_datetime.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        commit_hash=head_commit.hexsha,
        commit_stats=head_commit.stats.total,
    )


def fetch_git_repo(paths: Any, id_file: Path) -> Dict[str, str]:
    """Fetch the git repository.

    Args:
        paths: The paths of the git repository.
        id_file: The path to the id file.

    Returns:
        The metadata of the git repository.
    """
    git_command = get_git_command(id_file)

    if paths.local_path.is_dir():
        repo = Repo(paths.local_path)
        logger.debug(
            f"Repo {paths.local_path} already exists... Pulling changes from {repo.remotes.origin.url}"
        )
        with repo.git.custom_environment(GIT_SSH_COMMAND=git_command):
            repo.remotes.origin.pull()
    else:
        remote_url = giturlparse.parse(f"{paths.repo_path}").urls.get("ssh")

        logger.debug(f"Cloning {remote_url} to {paths.local_path}")
        repo = Repo.clone_from(
            remote_url, paths.local_path, env=dict(GIT_SSH_COMMAND=git_command)
        )
    return fetch_repo_metadata(repo)


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
            # "extra",
            # "abbr",
            # "attr_list",
            # "def_list",
            # "fenced_code",
            # "footnotes",
            # "md_in_html",
            # "admonition",
            # "legacy_attrs",
            # "legacy_em",
            # "meta",
            # "nl2br",
            # "sane_lists",
            # "smarty",
            "toc",
            # "wikilinks",
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
    cleaned_document = frontmatter.loads(cleaned_document).content
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
