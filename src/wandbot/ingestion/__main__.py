import os

from wandbot.ingestion import prepare_data, preprocess_data, vectorstores
from wandbot.ingestion.report import create_ingestion_report
from wandbot.utils import get_logger
from wandbot.configs.ingestion_config import IngestionConfig

ingestion_config = IngestionConfig()

logger = get_logger(__name__)


def main():
    project = ingestion_config.wandb_project
    entity = ingestion_config.wandb_entity

    raw_artifact = prepare_data.load(project, entity)
    preprocessed_artifact = preprocess_data.load(project, entity, raw_artifact)
    vectorstore_artifact = vectorstores.load(
        project, entity, preprocessed_artifact
    )

    create_ingestion_report(project, entity, raw_artifact, vectorstore_artifact)
    print(vectorstore_artifact)


if __name__ == "__main__":
    main()
