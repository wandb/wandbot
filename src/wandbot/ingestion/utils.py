import json
import pathlib
import re
import subprocess
from typing import Dict, List

import giturlparse
import markdown
import markdownify
import pandas as pd
from bs4 import BeautifulSoup, Comment
from git import Repo
from langchain.document_loaders import NotebookLoader
from langchain.document_loaders.notebook import remove_newlines
from langchain.schema import Document

from wandbot.utils import get_logger

logger = get_logger(__name__)


def get_git_command(id_file):
    assert id_file.is_file()

    git_command = f"ssh -v -i /{id_file}"
    return git_command


def fetch_git_remote_hash(repo_url=None, id_file=None):
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
    cell: dict, include_outputs: bool, max_output_length: int, traceback: bool
) -> str:
    """Combine cells information in a readable format ready to be used."""
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


class WandbNotebookLoader(NotebookLoader):
    """Loader that loads .ipynb notebook files in wandb examples."""

    def load(
        self,
    ) -> List[Document]:
        """Load documents."""
        p = pathlib.Path(self.file_path)

        with open(p, encoding="utf8") as f:
            d = json.load(f)

        data = pd.json_normalize(d["cells"])
        filtered_data = data[["cell_type", "source", "outputs"]]
        if self.remove_newline:
            filtered_data = filtered_data.applymap(remove_newlines)

        text = filtered_data.apply(
            lambda x: concatenate_cells(
                x, self.include_outputs, self.max_output_length, self.traceback
            ),
            axis=1,
        ).str.cat(sep=" ")

        metadata = {"source": str(p)}

        return [Document(page_content=text, metadata=metadata)]


EXTENSION_MAP = {
    ".py": "python",
    ".ipynb": "python",
    ".md": "markdown",
    ".js": "javascript",
    ".ts": "typescript",
}


def convert_contents_to_soup(contents):
    markdown_document = markdown.markdown(
        contents,
        extensions=[
            "extra",
            "abbr",
            "attr_list",
            "def_list",
            "fenced_code",
            "footnotes",
            "md_in_html",
            "admonition",
            "legacy_attrs",
            "legacy_em",
            "meta",
            "nl2br",
            "sane_lists",
            "smarty",
            "toc",
            "wikilinks",
        ],
    )
    soup = BeautifulSoup(markdown_document, "html.parser")
    return soup


def clean_soup(soup):
    for img_tag in soup.find_all("img", src=True):
        img_tag.extract()
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    for p_tag in soup.find_all("p"):
        if not p_tag.text.strip():
            p_tag.decompose()
    return soup


def clean_contents(contents):
    soup = convert_contents_to_soup(contents)
    soup = clean_soup(soup)
    cleaned_document = markdownify.MarkdownConverter(heading_style="ATX").convert_soup(
        soup
    )
    cleaned_document = cleaned_document.replace("![]()", "\n")
    cleaned_document = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cleaned_document)

    return cleaned_document