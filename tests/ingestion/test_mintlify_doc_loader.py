import json

from wandbot.configs.ingestion_config import DataSource, DataStoreConfig
from wandbot.ingestion.prepare_data import DocodileDataLoader


def make_doc_loader(
    tmp_path,
    include_only_published_docs: bool = False,
    exclude_patterns: list[str] | None = None,
):
    config = DataStoreConfig(
        name="english_documentation",
        source_type="documentation",
        data_source=DataSource(
            local_path=tmp_path,
            remote_path="https://docs.wandb.ai/",
            base_path="",
            file_patterns=["*.md", "*.mdx"],
            include_only_published_docs=include_only_published_docs,
            exclude_patterns=exclude_patterns or [],
        ),
        language="en",
    )
    return DocodileDataLoader(config)


def test_generate_site_url_uses_slug_frontmatter(tmp_path):
    loader = make_doc_loader(tmp_path)
    page = tmp_path / "models" / "registry.mdx"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "---\nslug: /models/model-registry/\ndescription: Model registry docs\n---\n# Registry\n"
    )

    site_url = loader.generate_site_url(tmp_path, page)

    assert site_url == "https://docs.wandb.ai/models/model-registry"


def test_generate_site_url_maps_index_to_directory_route(tmp_path):
    loader = make_doc_loader(tmp_path)
    page = tmp_path / "platform" / "index.mdx"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("# Platform\n")

    site_url = loader.generate_site_url(tmp_path, page)

    assert site_url == "https://docs.wandb.ai/platform"


def test_get_local_paths_filters_unpublished_files_with_docs_json(tmp_path):
    loader = make_doc_loader(tmp_path, include_only_published_docs=True)
    (tmp_path / "docs.json").write_text(
        json.dumps(
            {
                "tabs": [
                    {
                        "tab": "Docs",
                        "pages": [
                            "platform/index",
                            {"group": "Models", "pages": ["models/registry"]},
                        ],
                    }
                ]
            }
        )
    )
    published_index = tmp_path / "platform" / "index.mdx"
    published_page = tmp_path / "models" / "registry.mdx"
    unpublished_page = tmp_path / "README.md"
    published_index.parent.mkdir(parents=True, exist_ok=True)
    published_page.parent.mkdir(parents=True, exist_ok=True)
    published_index.write_text("# Platform\n")
    published_page.write_text("# Registry\n")
    unpublished_page.write_text("# Internal readme\n")

    local_paths = loader._get_local_paths()
    relative_paths = {
        local_path.relative_to(tmp_path).as_posix() for local_path in local_paths
    }

    assert relative_paths == {"platform/index.mdx", "models/registry.mdx"}


def test_get_local_paths_excludes_locale_paths(tmp_path):
    loader = make_doc_loader(
        tmp_path,
        exclude_patterns=["ja/**", "ko/**", "ja.mdx", "ko.mdx"],
    )
    (tmp_path / "ja.mdx").write_text("# Japanese landing page\n")
    japanese_page = tmp_path / "ja" / "guides" / "intro.mdx"
    korean_page = tmp_path / "ko" / "guides" / "intro.mdx"
    english_page = tmp_path / "guides" / "intro.mdx"
    japanese_page.parent.mkdir(parents=True, exist_ok=True)
    korean_page.parent.mkdir(parents=True, exist_ok=True)
    english_page.parent.mkdir(parents=True, exist_ok=True)
    japanese_page.write_text("# Japanese page\n")
    korean_page.write_text("# Korean page\n")
    english_page.write_text("# English page\n")

    local_paths = loader._get_local_paths()
    relative_paths = {
        local_path.relative_to(tmp_path).as_posix() for local_path in local_paths
    }

    assert relative_paths == {"guides/intro.mdx"}
