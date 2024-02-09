from typing import List, Optional

from llama_index import QueryBundle
from llama_index.postprocessor import BaseNodePostprocessor
from llama_index.schema import NodeWithScore

from wandbot.utils import create_no_result_dummy_node, get_logger

logger = get_logger(__name__)


class LanguageFilterPostprocessor(BaseNodePostprocessor):
    """Language-based Node processor."""

    languages: List[str] = ["en", "python"]
    min_result_size: int = 10

    @classmethod
    def class_name(cls) -> str:
        return "LanguageFilterPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""

        new_nodes = []
        for node in nodes:
            if node.metadata["language"] in self.languages:
                new_nodes.append(node)

        if len(new_nodes) < self.min_result_size:
            return new_nodes + nodes[: self.min_result_size - len(new_nodes)]

        return new_nodes


class MetadataPostprocessor(BaseNodePostprocessor):
    """Metadata-based Node processor."""

    min_result_size: int = 10
    include_tags: List[str] | None = None
    exclude_tags: List[str] | None = None

    @classmethod
    def class_name(cls) -> str:
        return "MetadataPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if not self.include_tags and not self.exclude_tags:
            return nodes
        new_nodes = []
        for node in nodes:
            normalized_tags = [
                tag.lower().strip() for tag in node.metadata["tags"]
            ]
            if self.include_tags:
                normalized_include_tags = [
                    tag.lower().strip() for tag in self.include_tags
                ]
                if not set(normalized_include_tags).issubset(
                    set(normalized_tags)
                ):
                    continue
            if self.exclude_tags:
                normalized_exclude_tags = [
                    tag.lower().strip() for tag in self.exclude_tags
                ]
                if set(normalized_exclude_tags).issubset(set(normalized_tags)):
                    continue
            new_nodes.append(node)
        if len(new_nodes) < self.min_result_size:
            dummy_node = create_no_result_dummy_node()
            new_nodes.extend(
                [dummy_node] * (self.min_result_size - len(new_nodes))
            )
        return new_nodes
