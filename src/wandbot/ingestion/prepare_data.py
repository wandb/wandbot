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

import datetime
import json
import logging
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
    IngestionConfig,
    SDKCodeStoreConfig,
    SDKTestsStoreConfig,
    WandbEduCodeStoreConfig,
    WeaveCodeStoreConfig,
    WeaveDocStoreConfig,
    DataSource,
)
from wandbot.ingestion.utils import (
    clean_contents,
    extract_frontmatter,
    fetch_git_repo,
)
from wandbot.schema.document import Document
from wandbot.utils import get_logger

logger = get_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

ingestion_config = IngestionConfig()


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
        local_paths = []
        search_base = (
            self.config.data_source.local_path
            / self.config.data_source.base_path
        )
        logger.info(f"Constructed search base path: {search_base.resolve()}")
        if not search_base.is_dir():
            logger.warning(f"Search base directory does not exist or is not a directory: {search_base}")
            return []
        logger.info(f"Searching for file patterns {self.config.data_source.file_patterns} in {search_base}")
        for file_pattern in self.config.data_source.file_patterns:
            # Add logging for count per pattern
            found_files = list(search_base.rglob(file_pattern))
            logger.info(f"Found {len(found_files)} files matching pattern '{file_pattern}' recursively in {search_base}")
            local_paths.extend(found_files)
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
        chapter = ""
        slug = ""
        file_loc = ""
        file_name = file_path.stem

        try:
            if relative_path.parts[0] == "guides":
                chapter = "guides"
                slug = self.extract_slug((base_path / "guides") / "_index.md")
                file_loc = file_path.relative_to((base_path / "guides")).parent
            elif relative_path.parts[0] == "ref":
                chapter = "ref"
                slug = self.extract_slug((base_path / "ref") / "_index.md")
                file_loc = file_path.relative_to((base_path / "ref")).parent
            elif relative_path.parts[0] == "tutorials":
                chapter = "tutorials"
                slug = self.extract_slug((base_path / "tutorials") / "_index.md")
                file_loc = file_path.relative_to((base_path / "tutorials")).parent
            else:
                # Fallback or handle other top-level directories if necessary
                chapter = relative_path.parts[0] if relative_path.parts else ""
                file_loc = relative_path.parent if len(relative_path.parts) > 1 else ""

            if file_path.name in ("_index.md",):
                file_name = ""

        except Exception as e:
            logger.debug(
                f"Failed to extract slug for URL generation from {file_path} due to frontmatter error: {e}. Using relative path for URL."
            )
            # Fallback logic: use relative path directly if slug extraction fails
            chapter = relative_path.parts[0] if relative_path.parts else ""
            file_loc = relative_path.parent if len(relative_path.parts) > 1 else ""
            file_name = file_path.stem if file_path.name not in ("_index.md",) else ""

        site_relative_path = os.path.join(chapter, slug, file_loc, file_name)
        # Clean up potential double slashes or leading/trailing slashes from fallback paths
        site_relative_path = "/".join(filter(None, str(site_relative_path).split("/")))

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
        if self.config.data_source.is_git_repo:
            self.metadata = fetch_git_repo(
                self.config.data_source, self.config.data_source.git_id_file
            )

        local_paths = self._get_local_paths()
        logger.info(
            f"Found {len(local_paths)} potential document files for {self.config.name}"
        )
        document_files = {
            local_path: self.generate_site_url(
                self.config.data_source.local_path / self.config.data_source.base_path,
                local_path,
            )
            for local_path in local_paths
        }

        for f_name in document_files:
            logger.debug(f"Processing document file: {f_name}")
            try:
                # Load the document content first
                document = TextLoader(str(f_name)).load()[0]
                contents = document.page_content

                # Clean contents (already made robust)
                document.page_content = clean_contents(contents)

                # Set basic metadata
                document.metadata["file_type"] = os.path.splitext(str(f_name))[-1]
                source_url = document_files[f_name]
                document.metadata["source"] = source_url
                document.metadata["language"] = self.config.language
                document.metadata["source_type"] = self.config.source_type
                logger.debug(f"Generated source URL: {source_url} for file: {f_name}")

                # Attempt to extract description (handles frontmatter errors)
                try:
                    document.metadata["description"] = self.extract_description(f_name)
                except Exception as e_desc:
                    logger.warning(
                        f"Failed to extract description from {f_name} due to: {e_desc}. Setting empty description."
                    )
                    document.metadata["description"] = ""

                # Attempt to extract tags (handles frontmatter errors if underlying functions use it)
                try:
                    # Assuming extract_tags might rely on data derived from frontmatter indirectly
                    # or could fail for other reasons. If it directly uses frontmatter, this is necessary.
                    document.metadata["tags"] = self.extract_tags(source_url)
                except Exception as e_tags:
                    logger.warning(
                        f"Failed to extract tags for {f_name} (source: {source_url}) due to: {e_tags}. Setting default tags."
                    )
                    document.metadata["tags"] = [
                        "Documentation"
                    ]  # Or provide a suitable default

                yield document

            except Exception as e_load:
                # Catch broader errors during file loading or initial processing
                logger.warning(
                    f"Failed to load or perform basic processing for documentation {f_name} due to: {e_load}",
                    exc_info=True,
                )
        logger.info(
            f"Processed {len(document_files)} document files for {self.config.name}"
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
        if self.config.data_source.is_git_repo:
            self.metadata = fetch_git_repo(
                self.config.data_source, self.config.data_source.git_id_file
            )

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
                logger.warning(f"Failed to load code in {f_name} with error {e}")


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
                    md_content += f"[{child['children'][0]['text']}]({child['url']})"
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
                                    md_content += f"`{text_block.get('text', '')}`"
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
        df = pd.json_normalize(reports_metadata, "tagIDs", ["id", "authors"]).rename(
            columns={0: "tagID", "id": "reportID"}
        )

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

        logger.debug(f"Before filtering, there are {len(report_ids_df)} reports")

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

        # Ensure the target directory exists before saving
        target_path = self.config.data_source.local_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        reports_df.to_json(
            target_path, lines=True, orient="records"
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
                    r"\ ", " "
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
                content = self.clean_invalid_unicode_escapes(row_dict["content"])
                content = content.encode("raw_unicode_escape").decode("unicode_escape")

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
                    + self.extract_tags(parsed_row["source"], parsed_row["content"]),
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
            logging.info("Identified Weave documentation loader.")
        else:
            source_type = "wandb_documentation"
            logging.info("Identified W&B documentation loader.")

    loader_class = SOURCE_TYPE_TO_LOADER_MAP.get(source_type)
    if loader_class:
        logging.info(
            f"Using loader {loader_class.__name__} for source type {source_type}"
        )
        return loader_class(config)
    else:
        logging.error(f"No loader found for source type {source_type}")
        raise ValueError(f"No loader found for source type {source_type}")


def load_from_config(config: DataStoreConfig) -> pathlib.Path:
    logging.info(f"Starting data loading for config: {config.name}")
    try:
        loader = get_loader_from_config(config)
        docstore_dir = loader.config.docstore_dir
        logging.info(f"Using docstore directory: {docstore_dir}")
        docstore_dir.mkdir(parents=True, exist_ok=True)

        config_path = docstore_dir / "config.json"
        logging.info(f"Writing config to {config_path}")
        with config_path.open("w") as f:
            f.write(loader.config.model_dump_json())

        documents_path = docstore_dir / "documents.jsonl"
        logging.info(f"Starting document loading into {documents_path}")
        doc_count = 0
        with documents_path.open("w") as f:
            for document in loader.load():
                document_json = {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                f.write(json.dumps(document_json) + "\n")
                doc_count += 1
        logging.info(f"Finished loading {doc_count} documents into {documents_path}")

        metadata_path = docstore_dir / "metadata.json"
        logging.info(f"Writing metadata to {metadata_path}")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w") as f:
            json.dump(loader.metadata, f)

        logging.info(f"Successfully completed data loading for config: {config.name}")
        return docstore_dir
    except Exception as e:
        logging.error(
            f"Failed loading data for config {config.name}: {e}", exc_info=True
        )
        # Reraise the exception so the multiprocessing pool knows about the failure
        raise


def load(
    project: str | None = ingestion_config.wandb_project,
    entity: str | None = ingestion_config.wandb_entity,
    result_artifact_name: str = "raw_data",
) -> str:
    """Loads and prepares data for the Wandbot ingestion system.

    This function orchestrates the data loading process for various data sources
    defined in the configuration. It initializes a Weights & Biases run,
    loads data using the appropriate data loaders, saves the loaded documents
    to JSONL files, and logs the resulting data as a W&B artifact.

    Args:
        project: The Weights & Biases project name.
        entity: The Weights & Biases entity (user or team) name.
        result_artifact_name: The name for the resulting W&B artifact.

    Returns:
        The path to the logged artifact.
    """
    logging.info(f"Starting Wandbot data ingestion for {entity}/{project}")
    run = wandb.init(project=project, entity=entity, job_type="prepare_dataset")
    logging.info(f"Initialized Wandb run: {run.url}")

    # Generate a timestamp string for this run
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_cache_root = ingestion_config.cache_dir / timestamp_str
    logging.info(f"Using timestamped cache root: {timestamped_cache_root}")

    # Define all configurations
    all_configs: List[DataStoreConfig] = [
        DocodileEnglishStoreConfig(),
        DocodileJapaneseStoreConfig(),
        DocodileKoreanStoreConfig(),
        WeaveDocStoreConfig(),
        WeaveCodeStoreConfig(),
        SDKCodeStoreConfig(),
        SDKTestsStoreConfig(),
        ExampleCodeStoreConfig(),
        ExampleNotebookStoreConfig(),
        WandbEduCodeStoreConfig(),
        FCReportsStoreConfig(),
    ]

    # Filter for the specific config we want to debug
    # configs_to_process = [
    #     config
    #     for config in all_configs
    #     if isinstance(config, DocodileEnglishStoreConfig)
    # ]
    configs_to_process = all_configs

    # Update docstore_dir for each config to include the timestamp directory
    for config in configs_to_process:
        original_docstore_name = config.docstore_dir.name
        # Get the default 'raw_data' dir name from the DataSource definition
        raw_data_dir_name = DataSource().cache_dir.name
        config.docstore_dir = (
            timestamped_cache_root / raw_data_dir_name / original_docstore_name
        )
        logging.debug(
            f"Updated docstore_dir for {config.name} to {config.docstore_dir}"
        )

    # Use multiprocessing to load data in parallel
    num_processes = max(1, cpu_count() // 2)
    pool = Pool(num_processes)

    results = pool.map(load_from_config, configs_to_process)

    pool.close()
    pool.join()

    artifact = wandb.Artifact(
        result_artifact_name,
        type="dataset",
        description="Raw documents for wandbot",
    )
    logging.info(f"Created Wandb Artifact: {result_artifact_name}")

    completed_count = 0
    total_configs = len(configs_to_process)
    added_to_artifact = []

    for docstore_path in results:
        try:
            # The docstore_path is already in the correct timestamped location.
            # No need to rename locally.
            # We need the original config name for the artifact structure.
            # Assuming the structure is timestamped_cache_root / 'raw_data' / config_name
            artifact_entry_name = docstore_path.name  # e.g., English_Documentation
            artifact_base_dir = docstore_path.parent.name  # e.g., raw_data
            artifact_full_name = f"{artifact_base_dir}/{artifact_entry_name}"

            # The path already includes the timestamp. Rename for artifact clarity (remove run_id if needed).
            # Keep the original logic for adding to artifact for now.
            # unique_local_path = docstore_path.parent / f"{docstore_path.name}_{run.id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            # logging.info(f"Renaming local cache {docstore_path} to {unique_local_path}")
            # docstore_path.rename(unique_local_path)

            logging.info(
                f"Adding directory {docstore_path} to artifact {result_artifact_name} as {artifact_full_name}"
            )
            artifact.add_dir(
                str(docstore_path),  # Use the final timestamped path
                name=artifact_full_name,  # Use structure like 'raw_data/English_Documentation'
            )
            added_to_artifact.append(artifact_full_name)
            completed_count += 1
            logging.info(
                f"Progress: {completed_count}/{total_configs} configurations processed and added."
            )
        except Exception as e:
            logging.error(
                f"Failed to add directory {docstore_path} to artifact: {e}",
                exc_info=True,
            )

    logging.info(f"Successfully added directories to artifact: {added_to_artifact}")

    if completed_count < total_configs:
        logging.warning(
            f"Only {completed_count} out of {total_configs} configurations completed successfully and added to artifact."
        )
    else:
        logging.info(
            f"All {total_configs} configurations processed successfully and added to artifact."
        )

    logging.info(f"Logging artifact {result_artifact_name} to Wandb.")
    run.log_artifact(artifact)
    logging.info("Artifact logged successfully.")
    run.finish()
    logging.info(f"Wandb run finished: {run.url}")
    artifact_path = f"{entity}/{project}/{result_artifact_name}:latest"
    logging.info(f"Data ingestion finished. Result artifact: {artifact_path}")
    return artifact_path
