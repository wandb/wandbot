"""
This module contains classes and functions for preparing data in the Wandbot ingestion system.

The module includes the following classes:
- `DataLoader`: A base class for data loaders that provides a base implementation for lazy loading of documents.
- `DocodileDataLoader`: A data loader specifically designed for Docodile documents.
- `CodeDataLoader`: A data loader for code documents.

The module also includes the following functions:
- `load`: Loads and prepares data for the Wandbot ingestion system.

Typical usage example:

    load(project="my_project", entity="my_entity", result_artifact_name="raw_dataset")
"""

import json
import os
import pathlib
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterator, List
from urllib.parse import urljoin, urlparse

import nbformat
import pandas as pd
from google.cloud import bigquery
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.base import BaseLoader
from nbconvert import MarkdownExporter
from nbformat.validator import normalize

import wandb
from wandbot.configs.ingestion_config import (
    DataStoreConfig,
    DocodileEnglishStoreConfig,
    DocodileJapaneseStoreConfig,
    DocodileKoreanStoreConfig,
    ExampleCodeStoreConfig,
    ExampleNotebookStoreConfig,
    FCReportsStoreConfig,
    SDKCodeStoreConfig,
    SDKTestsStoreConfig,
    WandbEduCodeStoreConfig,
    WeaveCodeStoreConfig,
    WeaveDocStoreConfig,
    WeaveExamplesStoreConfig,
)
from wandbot.ingestion.utils import (
    clean_contents,
    extract_frontmatter,
    fetch_git_repo,
)
from wandbot.utils import get_logger
from wandbot.schema.document import Document

logger = get_logger(__name__)


class DataLoader(BaseLoader):
    """A base class for data loaders.

    This class provides a base implementation for lazy loading of documents.
    Subclasses should implement the `lazy_load` method to define the specific
    loading behavior.
    """

    def __init__(self, config: DataStoreConfig):
        """Initializes the DataLoader instance.

        Args:
            config: The configuration for the data store.
        """
        self.config: DataStoreConfig = config
        self.metadata: Dict[str, Any] = {}

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Documents.

        This method should be implemented by subclasses to define the specific
        loading behavior.

        Returns:
            An iterator of Document objects.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )

    def load(self):
        """Loads the documents.

        Returns:
            A list of Document objects.
        """
        documents = list(self.lazy_load())
        self.metadata.update({"num_documents": len(documents)})
        return documents

    def _get_local_paths(self):
        if self.config.data_source.is_git_repo:
            self.metadata = fetch_git_repo(
                self.config.data_source, self.config.data_source.git_id_file
            )

        local_paths = []
        for file_pattern in self.config.data_source.file_patterns:
            local_paths.extend(
                list(
                    (
                        self.config.data_source.local_path
                        / self.config.data_source.base_path
                    ).rglob(file_pattern)
                )
            )
        return local_paths


class DocodileDataLoader(DataLoader):
    """A data loader for Docodile documents.

    This class provides a data loader specifically designed for Docodile documents.
    It implements the lazy_load method to define the loading behavior.

    Attributes:
        config: The configuration for the data store.
    """

    @staticmethod
    def extract_slug(file_path: pathlib.Path) -> str:
        """Extracts the slug from a file.

        Args:
            file_path: The path to the file.

        Returns:
            The extracted slug.
        """

        meta = extract_frontmatter(file_path)
        output = meta.get("slug", "")
        return output

    @staticmethod
    def extract_description(file_path: pathlib.Path) -> str:
        """Extracts the description from a file.

        Args:
            file_path: The path to the file.

        Returns:
            The extracted description.
        """
        meta = extract_frontmatter(file_path)
        output = meta.get("description", "")
        return output

    @staticmethod
    def extract_tags(source_url: str) -> List[str]:
        """Extracts the tags from a source url.

        Args:
            source_url: The URL of the file.

        Returns:
            The extracted tags.
        """
        parts = list(filter(lambda x: x, urlparse(source_url).path.split("/")))
        parts_mapper = {
            "ref": ["API Reference", "Technical Specifications"],
            "guides": ["Guides"],
            "tutorials": ["Tutorials"],
        }
        tags = []
        for part in parts:
            if part in parts_mapper:
                tags.extend(parts_mapper.get(part, []))
            else:
                part = part.replace("-", " ")
                tags.append(part)
        tags = [tag.split(".")[0] for tag in tags]
        tags = list(set([tag.title() for tag in tags]))
        return tags + ["Documentation"]

    def generate_site_url(
        self, base_path: pathlib.Path, file_path: pathlib.Path
    ) -> str:
        """Generates the site URL for a file.

        Args:
            base_path: The base path of the file.
            file_path: The path to the file.

        Returns:
            The generated site URL.
        """
        relative_path = file_path.relative_to(base_path)
        if relative_path.parts[0] == "guides":
            chapter = "guides"
            slug = self.extract_slug((base_path / "guides") / "intro.md")
            file_loc = file_path.relative_to((base_path / "guides")).parent
        elif relative_path.parts[0] == "ref":
            chapter = "ref"
            slug = self.extract_slug((base_path / "ref") / "README.md")
            file_loc = file_path.relative_to((base_path / "ref")).parent
        elif relative_path.parts[0] == "tutorials":
            chapter = "tutorials"
            slug = self.extract_slug(
                (base_path / "tutorials") / "intro_to_tutorials.md"
            )
            file_loc = file_path.relative_to((base_path / "tutorials")).parent
        else:
            chapter = ""
            slug = ""
            file_loc = ""

        file_name = file_path.stem
        if file_path.name in ("intro.md", "README.md", "intro_to_tutorials.md"):
            file_name = ""
        site_relative_path = os.path.join(chapter, slug, file_loc, file_name)
        site_url = urljoin(
            str(self.config.data_source.remote_path), str(site_relative_path)
        )
        if "other/" in site_url:
            site_url = site_url.replace("other/", "")

        return site_url

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Docodile documents.

        This method implements the lazy loading behavior for Docodile documents.

        Yields:
            A Document object.
        """
        local_paths = self._get_local_paths()
        document_files = {
            local_path: self.generate_site_url(
                self.config.data_source.local_path
                / self.config.data_source.base_path,
                local_path,
            )
            for local_path in local_paths
        }

        for f_name in document_files:
            try:
                document = TextLoader(f_name).load()[0]
                contents = document.page_content
                document.page_content = clean_contents(contents)
                document.metadata["file_type"] = os.path.splitext(
                    document.metadata["source"]
                )[-1]
                document.metadata["source"] = document_files[
                    pathlib.Path(document.metadata["source"])
                ]
                document.metadata["language"] = self.config.language
                document.metadata["description"] = self.extract_description(
                    f_name
                )
                document.metadata["tags"] = self.extract_tags(
                    document.metadata["source"]
                )
                document.metadata["source_type"] = self.config.source_type
                yield document
            except Exception as e:
                logger.warning(
                    f"Failed to load documentation {f_name} due to {e}"
                )


class WeaveDocsDataLoader(DocodileDataLoader):
    def generate_site_url(
        self, base_path: pathlib.Path, file_path: pathlib.Path
    ) -> str:
        chapter = ""
        slug = ""
        file_loc = ""

        file_name = file_path.stem
        if file_path.name in ("introduction.md",):
            file_name = ""
        site_relative_path = os.path.join(chapter, slug, file_loc, file_name)
        site_url = urljoin(
            str(self.config.data_source.remote_path), str(site_relative_path)
        )
        return site_url


class CodeDataLoader(DataLoader):
    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for code documents.

        This method implements the lazy loading behavior for code documents.

        Yields:
            A Document object.
        """
        local_paths = self._get_local_paths()

        paths = list(local_paths)
        local_paths = list(map(lambda x: str(x), paths))
        local_path_parts = list(map(lambda x: x.parts, paths))
        examples_idx = list(
            map(
                lambda x: x.index(self.config.data_source.local_path.stem),
                local_path_parts,
            )
        )
        remote_paths = list(
            map(
                lambda x: "/".join(x[1][x[0] + 1 :]),
                zip(examples_idx, local_path_parts),
            )
        )
        remote_paths = list(
            map(
                lambda x: f"{self.config.data_source.remote_path}{x}",
                remote_paths,
            )
        )
        document_files = dict(zip(local_paths, remote_paths))

        for f_name in document_files:
            try:
                if os.path.splitext(f_name)[-1] == ".ipynb":
                    document = TextLoader(f_name).load()[0]
                    contents = document.page_content
                    notebook = nbformat.reads(contents, as_version=4)
                    _, notebook = normalize(
                        notebook, version=4, strip_invalid_metadata=True
                    )  # Normalize the notebook
                    md_exporter = MarkdownExporter(template="classic")
                    (body, resources) = md_exporter.from_notebook_node(notebook)
                    cleaned_body = clean_contents(body)
                    document.page_content = cleaned_body
                    document.metadata["source_type"] = (
                        "notebook"
                        if self.config.source_type == "code"
                        else self.config.source_type
                    )
                elif os.path.splitext(f_name)[-1] == ".md":
                    document = TextLoader(f_name).load()[0]
                    contents = document.page_content
                    cleaned_body = clean_contents(contents)
                    document.page_content = cleaned_body
                    document.metadata["source_type"] = (
                        "markdown"
                        if self.config.source_type == "code"
                        else self.config.source_type
                    )
                else:
                    document = TextLoader(f_name).load()[0]
                    document.metadata["source_type"] = (
                        "code"
                        if self.config.source_type == "code"
                        else self.config.source_type
                    )

                document.metadata["file_type"] = os.path.splitext(
                    document.metadata["source"]
                )[-1]
                document.metadata["source"] = document_files[
                    document.metadata["source"]
                ]
                yield document
            except Exception as e:
                logger.warning(
                    f"Failed to load code in {f_name} with error {e}"
                )


class FCReportsDataLoader(DataLoader):
    def get_reports_ids(self):
        client = bigquery.Client(project=self.config.data_source.remote_path)

        query = """
            SELECT DISTINCT
                lower(REGEXP_REPLACE(COALESCE(
                REGEXP_EXTRACT(report_path, "(--[Vv]ml[^/?]+)"),
                REGEXP_EXTRACT(report_path, "(/[Vv]ml[^/?]+)"), 
                REGEXP_EXTRACT(report_path, "([Vv]ml[^/?]+)")),
                "^[/-]+", "")) as reportID
            FROM 
                analytics.dim_reports
            WHERE
                is_public
            """

        report_ids_df = client.query(query).to_dataframe()
        return report_ids_df

    def get_reports_from_bigquery(self, report_ids: str, created_after=None):
        client = bigquery.Client(project=self.config.data_source.remote_path)

        query = f"""
        SELECT 
           created_at, 
           description,
           display_name, 
           is_public, 
           created_using,
           report_id, 
           project_id,
           type,
           name,
           report_path, 
           showcased_at, 
           spec, 
           stars_count, 
           user_id, 
           view_count
        FROM 
            analytics.dim_reports
        WHERE
            is_public
            and LOWER(REGEXP_EXTRACT(report_path, r'reports/--(.*)')) in ({report_ids})
        """

        # # AND showcased_at IS NOT NULL

        if created_after:
            query += f"AND created_at >= '{created_after}'"

        reports_df = client.query(query).to_dataframe()
        return reports_df

    def get_fully_connected_spec(self):
        client = bigquery.Client(project=self.config.data_source.remote_path)

        query = """
            SELECT *
            FROM `analytics.stg_mysql_views`
            WHERE type = 'fullyConnected' 
        """

        fc_spec = client.query(query).to_dataframe()
        return fc_spec

    def convert_block_to_markdown(self, block):
        """
        Converts a single content block to its Markdown representation.
        """
        md_content = ""

        # Handle different types of blocks
        if block["type"] == "paragraph":
            for child in block["children"]:
                if "type" in child:
                    if (
                        child["type"] == "block-quote"
                        or child["type"] == "callout-block"
                    ):
                        md_content += self.convert_block_to_markdown(child)
                    elif "url" in child:
                        md_content += (
                            f"[{child['children'][0]['text']}]({child['url']})"
                        )
                    elif "inlineCode" in child and child["inlineCode"]:
                        md_content += f"`{child.get('text', '')}`"
                    else:
                        md_content += child.get("text", "")
                else:
                    md_content += child.get("text", "")
            md_content += "\n\n"

        elif block["type"] == "heading":
            md_content += "#" * block["level"] + " "
            for child in block["children"]:
                if "url" in child:
                    md_content += (
                        f"[{child['children'][0]['text']}]({child['url']})"
                    )
                else:
                    md_content += child.get("text", "")
            md_content += "\n\n"

        elif block["type"] == "list":
            for item in block["children"]:
                if item["type"] == "list-item":
                    md_content += "* "
                    for child in item["children"]:
                        if child.get("type") == "paragraph":
                            for text_block in child["children"]:
                                if (
                                    "inlineCode" in text_block
                                    and text_block["inlineCode"]
                                ):
                                    md_content += (
                                        f"`{text_block.get('text', '')}`"
                                    )
                                else:
                                    md_content += text_block.get("text", "")
                        else:
                            if "inlineCode" in child and child["inlineCode"]:
                                md_content += f"`{child.get('text', '')}`"
                            else:
                                md_content += child.get("text", "")
                    md_content += "\n"
            md_content += "\n"

        elif block["type"] == "code-block":
            md_content += "```\n"
            for line in block["children"]:
                md_content += line["children"][0].get("text", "") + "\n"
            md_content += "```\n\n"

        elif block["type"] == "block-quote" or block["type"] == "callout-block":
            md_content += "\n> "
            for child in block["children"]:
                md_content += child.get("text", "") + " "
            md_content += "\n\n"

        elif block["type"] == "horizontal-rule":
            md_content += "\n---\n"

        elif block["type"] == "latex":
            latex_content = block.get("content", "")
            md_content += f"$$\n{latex_content}\n$$\n\n"

        return md_content

    @staticmethod
    def extract_panel_group_text(spec_dict):
        # Initialize an empty list to store the content
        content = ""

        # Extract content under "panelGroups"
        for block in spec_dict.get("panelGroups", []):
            # content_list.append(block.get('content', ''))
            content += block.get("content", "") + "\n"

        return content

    def spec_to_markdown(self, spec, report_id, report_name):
        spec_dict = json.loads(spec)
        markdown_text = ""
        buggy_report_id = None
        spec_type = ""
        is_buggy = False

        if "panelGroups" in spec:
            markdown_text = self.extract_panel_group_text(spec_dict)
            spec_type = "panelgroup"

        else:
            try:
                for i, block in enumerate(spec_dict["blocks"]):
                    try:
                        markdown_text += self.convert_block_to_markdown(block)
                        spec_type = "blocks"
                    except Exception as e:
                        logger.debug(
                            f"Error converting block {i} in report_id: {report_id},"
                            f" {report_name}:\n{e}\nSPEC DICT:\n{spec_dict}\n"
                        )

                        markdown_text = ""
                        buggy_report_id = report_id
                        is_buggy = True
            except Exception as e:
                logger.debug(
                    f"Error finding 'blocks' in spec for report_id: {report_id}, {report_name} :\n{e}\n"
                )
                markdown_text = ""
                buggy_report_id = report_id
                is_buggy = True
        return markdown_text, buggy_report_id, is_buggy, spec_type

    @staticmethod
    def extract_fc_report_ids(fc_spec_df):
        fc_spec = json.loads(fc_spec_df["spec"].values[0])
        reports_metadata = fc_spec["reportIDsWithTagV2IDs"]
        df = pd.json_normalize(
            reports_metadata, "tagIDs", ["id", "authors"]
        ).rename(columns={0: "tagID", "id": "reportID"})

        # Drop duplicates on 'reportID' column
        df = df.drop_duplicates(subset="reportID")
        df["reportID"] = df["reportID"].str.lower()
        # drop the tagID column
        df = df.drop(columns=["tagID"])
        # convert authors to string, unpack it if it's a list
        df["authors"] = df["authors"].astype(str)
        return df

    def cleanup_reports_df(self, reports_df):
        markdown_ls = []
        buggy_report_ids = []
        spec_type_ls = []
        is_buggy_ls = []
        is_short_report_ls = []
        for idx, row in reports_df.iterrows():
            if row["spec"] is None or isinstance(row["spec"], float):
                logger.debug(idx)
                markdown_ls.append("spec-error")
                buggy_report_ids.append(row["report_id"])
                spec_type_ls.append("spec-error")
                is_buggy_ls.append(True)
                is_short_report_ls.append(True)

            else:
                (
                    markdown,
                    buggy_report_id,
                    is_buggy,
                    spec_type,
                ) = self.spec_to_markdown(
                    row["spec"], row["report_id"], row["display_name"]
                )

                markdown_ls.append(markdown)
                buggy_report_ids.append(buggy_report_id)
                spec_type_ls.append(spec_type)
                is_buggy_ls.append(is_buggy)

                # check if markdown has less than 100 characters
                if len(markdown) < 100:
                    is_short_report_ls.append(True)
                else:
                    is_short_report_ls.append(False)

        reports_df["markdown_text"] = markdown_ls
        reports_df["spec_type"] = spec_type_ls
        reports_df["is_buggy"] = is_buggy_ls
        reports_df["is_short_report"] = is_short_report_ls

        reports_df["content"] = (
            "\n# "
            + reports_df["display_name"].astype(str)
            + "\n\nDescription: "
            + reports_df["description"].astype(str)
            + "\n\nBody:\n\n"
            + reports_df["markdown_text"].astype(str)
        )

        reports_df["character_count"] = reports_df["content"].map(len)

        # reports_df["character_count"] = len(reports_df["content"])
        reports_df["source"] = "https://wandb.ai" + reports_df["report_path"]

        # tidy up the dataframe
        reports_df.drop(columns=["markdown_text"], inplace=True)
        reports_df.drop(columns=["spec"], inplace=True)
        reports_df.sort_values(by=["created_at"], inplace=True, ascending=False)
        reports_df.reset_index(drop=True, inplace=True)
        return reports_df

    def fetch_data(self):
        report_ids_df = self.get_reports_ids()
        fc_spec_df = self.get_fully_connected_spec()
        fc_ids_df = self.extract_fc_report_ids(fc_spec_df)

        logger.debug(
            f"Before filtering, there are {len(report_ids_df)} reports"
        )

        report_ids_df = report_ids_df.merge(
            fc_ids_df,
            how="inner",
            on="reportID",
        )

        logger.debug(
            f"After filtering, there are {len(report_ids_df)} FC report IDs to fetch"
        )

        # Pass report ids into a string for BigQuery query
        report_ids_str = ""
        for idx in report_ids_df["reportID"].values:
            report_ids_str += f"'{idx}',"
        report_ids_str = report_ids_str[:-1]
        reports_df = self.get_reports_from_bigquery(report_ids_str)

        reports_df["source"] = "https://wandb.ai" + reports_df["report_path"]
        reports_df["reportID"] = (
            reports_df["report_path"]
            .str.split("reports/--", n=1, expand=True)[1]
            .str.lower()
        )
        reports_df["description"] = reports_df["description"].fillna("")
        reports_df["display_name"] = reports_df["display_name"].fillna("")

        logger.debug(f"{len(reports_df)} Fully Connected Reports fetched")

        reports_df = self.cleanup_reports_df(reports_df)

        reports_df.to_json(
            self.config.data_source.local_path, lines=True, orient="records"
        )

        return self.config.data_source.local_path

    @staticmethod
    def clean_invalid_unicode_escapes(text):
        """
        clean up invalid unicode escape sequences
        """
        # List of common escape sequences to retain
        common_escapes = ["\\n", "\\t", "\\r", "\\\\", "\\'", '\\"']

        # Replace each uncommon escape sequence with a space or other character
        for i in range(256):
            escape_sequence = f"\\{chr(i)}"
            if escape_sequence not in common_escapes:
                text = text.replace(
                    escape_sequence, "  "
                )  # replace with a space or any character of your choice
                text = text.replace(
                    "\ ", " "
                )  # in case an invalid escape sequence was created above
        return text

    def parse_row(self, row):
        row_dict = json.loads(row)
        if (
            not (row_dict["is_short_report"] or row_dict["is_buggy"])
            and row_dict["character_count"] > 100
        ):
            try:
                content = (
                    repr(row_dict["content"])
                    .encode("raw_unicode_escape")
                    .decode("unicode_escape")
                )
            except UnicodeDecodeError:
                # fix escape characters with raw_unicode_escape
                content = self.clean_invalid_unicode_escapes(
                    row_dict["content"]
                )
                content = content.encode("raw_unicode_escape").decode(
                    "unicode_escape"
                )

            output = {
                "content": content,
                "source": row_dict["source"],
                "description": row_dict["description"],
            }

            return output

    def parse_data_dump(self, data_file):
        for row in open(data_file):
            parsed_row = self.parse_row(row)
            if parsed_row:
                yield parsed_row

    @staticmethod
    def extract_tags(source_url: str, report_content: str) -> List[str]:
        """Extracts the tags from a source url and the FC Report content.

        Args:
            source_url: The URL of the file.
            report_content: The content of the FC Report.

        Returns:
            The extracted tags.
        """
        parts = list(filter(lambda x: x, urlparse(source_url).path.split("/")))
        parts_mapper = {
            "ml-news": ["ml-news"],
            "gradient-dissent": ["gradient-dissent"],
            "event-announcement": ["event-announcements"],
            "events": ["events"],
            "announcements": ["announcements"],
            "launch-releases": ["launch-releases"],
        }
        tags = []
        for part in parts:
            if part in parts_mapper:
                tags.extend(parts_mapper.get(part, []))
            else:
                part = part.replace("-", " ")
                tags.append(part)
        tags = [tag.split(".")[0] for tag in tags]
        tags = list(set([tag.title() for tag in tags]))

        if ("wandb.log" or "wandb.init") in report_content:
            tags.append("Contains Wandb Code")

        return tags

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for code documents.

        This method implements the lazy loading behavior for report documents.

        Yields:
            A Document object.
        """
        data_dump_fame = self.fetch_data()
        for parsed_row in self.parse_data_dump(data_dump_fame):
            document = Document(
                page_content=parsed_row["content"],
                metadata={
                    "source": parsed_row["source"],
                    "source_type": self.config.source_type,
                    "file_type": ".md",
                    "description": parsed_row["description"],
                    "tags": ["Fully Connected", "Report"]
                    + self.extract_tags(
                        parsed_row["source"], parsed_row["content"]
                    ),
                },
            )
            yield document


SOURCE_TYPE_TO_LOADER_MAP = {
    "wandb_documentation": DocodileDataLoader,
    "weave_documentation": WeaveDocsDataLoader,
    "code": CodeDataLoader,
    "notebook": CodeDataLoader,
    "report": FCReportsDataLoader,
}


def get_loader_from_config(config: DataStoreConfig) -> DataLoader:
    """Get the DataLoader class based on the source type.

    Args:
        config: The configuration for the data store.

    Returns:
        The DataLoader class.
    """
    source_type = config.source_type
    if source_type == "documentation":
        if "weave" in config.name.lower():
            source_type = "weave_documentation"
        else:
            source_type = "wandb_documentation"

    return SOURCE_TYPE_TO_LOADER_MAP[source_type](config)


def load_from_config(config: DataStoreConfig) -> pathlib.Path:
    loader = get_loader_from_config(config)
    loader.config.docstore_dir.mkdir(parents=True, exist_ok=True)

    with (loader.config.docstore_dir / "config.json").open("w") as f:
        f.write(loader.config.model_dump_json())

    with (loader.config.docstore_dir / "documents.jsonl").open("w") as f:
        for document in loader.load():
            document_json = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            f.write(json.dumps(document_json) + "\n")
    with (loader.config.docstore_dir / "metadata.json").open("w") as f:
        json.dump(loader.metadata, f)
    return loader.config.docstore_dir


def load(
    project: str,
    entity: str,
    result_artifact_name: str = "raw_dataset",
) -> str:
    """Load and prepare data for the Wandbot ingestion system.

    This function initializes a Wandb run, creates an artifact for the prepared dataset,
    and loads and prepares data from different loaders. The prepared data is then saved
    in the docstore directory and added to the artifact.

    Args:
        project: The name of the Wandb project.
        entity: The name of the Wandb entity.
        result_artifact_name: The name of the result artifact. Default is "raw_dataset".

    Returns:
        The latest version of the prepared dataset artifact in the format
        "{entity}/{project}/{result_artifact_name}:latest".
    """
    run = wandb.init(project=project, entity=entity, job_type="prepare_dataset")
    artifact = wandb.Artifact(
        result_artifact_name,
        type="dataset",
        description="Raw documents for wandbot",
    )

    configs = [
        DocodileEnglishStoreConfig(),
        DocodileJapaneseStoreConfig(),
        DocodileKoreanStoreConfig(),
        ExampleCodeStoreConfig(),
        ExampleNotebookStoreConfig(),
        SDKCodeStoreConfig(),
        SDKTestsStoreConfig(),
        WeaveDocStoreConfig(),
        WeaveCodeStoreConfig(),
        WeaveExamplesStoreConfig(),
        WandbEduCodeStoreConfig(),
        FCReportsStoreConfig(),
    ]

    pool = Pool(cpu_count() - 1)
    results = pool.imap_unordered(load_from_config, configs)

    for docstore_path in results:
        artifact.add_dir(
            str(docstore_path),
            name=docstore_path.name,
        )
    run.log_artifact(artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"
