import logging

from wandbot.ingestion.config import (
    DocumentationStoreConfig,
    ExampleCodeStoreConfig,
    ExampleNotebookStoreConfig,
    GTMDataStoreConfig,
    SDKCodeStoreConfig,
    VectorIndexConfig,
)
from wandbot.ingestion.datastore import (
    CodeDataStore,
    DocumentationDataStore,
    GTMDataStore,
    VectorIndex,
)
from wandbot.ingestion.report import create_ingestion_report
from wandbot.ingestion.utils import save_dataset

logger = logging.getLogger(__name__)


def main():
    data_sources = [
        DocumentationDataStore(DocumentationStoreConfig()),
        CodeDataStore(ExampleCodeStoreConfig()),
        CodeDataStore(ExampleNotebookStoreConfig()),
        CodeDataStore(SDKCodeStoreConfig()),
        # CodeDataStore(WeaveCodeStoreConfig()),
        # ExtraDataStore(ExtraDataStoreConfig()),
        GTMDataStore(GTMDataStoreConfig()),
    ]
    vectorindex_config = VectorIndexConfig(wandb_project="wandb_docs_bot_dev")
    vector_index = VectorIndex(vectorindex_config)
    vector_index = vector_index.load(data_sources)
    vector_index.save()
    raw_dataset_artifact = save_dataset(data_sources)
    create_ingestion_report(vector_index, raw_dataset_artifact)


if __name__ == "__main__":
    main()
