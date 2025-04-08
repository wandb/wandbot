# src/wandbot/ingestion/config.py
from dataclasses import dataclass, field
import simple_parsing as sp
from typing import List

@dataclass
class IngestionRunConfig:
    """Command-line arguments for controlling the ingestion pipeline."""
    steps: List[str] = field(
        default_factory=lambda: ["prepare", "preprocess", "vectorstore", "report"]
    )
    """Steps to run: prepare, preprocess, vectorstore, report"""
    include_sources: List[str] = field(default_factory=list)
    """List of specific source names (from ingestion_config.py) to include. If empty, includes all (respecting excludes)."""
    exclude_sources: List[str] = field(default_factory=list)
    """List of specific source names to exclude. Applied after includes."""
    raw_data_artifact_name: str = "raw_data"
    """Override the default raw data artifact name."""
    preprocessed_data_artifact_name: str = "transformed_data"
    """Override the default preprocessed data artifact name."""
    vectorstore_artifact_name: str = "chroma_index"
    """Override the default vector store artifact name."""
    debug: bool = False
    """Run in debug mode: process only the first source and first 3 documents, append _debug to artifact names."""

def get_run_config() -> IngestionRunConfig:
    """Parses command line arguments for ingestion run configuration."""
    parser = sp.ArgumentParser(add_option_string_dash_variants=True)
    parser.add_arguments(IngestionRunConfig, dest="run_config")
    args = parser.parse_args()
    return args.run_config 