import logging

from wandbot.ingestion.config import (
    DocumentationStoreConfig,
    ExampleCodeStoreConfig,
    ExampleNotebookStoreConfig,
    ExtraDataStoreConfig,
    GTMDataStoreConfig,
    SDKCodeStoreConfig,
    VectorIndexConfig,
)
from wandbot.ingestion.datastore import (
    CodeDataStore,
    DocumentationDataStore,
    ExtraDataStore,
    GTMDataStore,
    VectorIndex,
)
from wandbot.ingestion.report import create_ingestion_report

logger = logging.getLogger(__name__)


def main():
    data_sources = [
        DocumentationDataStore(DocumentationStoreConfig()),
        CodeDataStore(ExampleCodeStoreConfig()),
        CodeDataStore(ExampleNotebookStoreConfig()),
        CodeDataStore(SDKCodeStoreConfig()),
        ExtraDataStore(ExtraDataStoreConfig()),
        GTMDataStore(GTMDataStoreConfig()),
    ]
    vectorindex_config = VectorIndexConfig(wandb_project="wandb_docs_bot_dev")
    vector_index = VectorIndex(vectorindex_config)
    vector_index = vector_index.load(data_sources)
    vector_index.save()
    create_ingestion_report(vector_index)


if __name__ == "__main__":
    main()
