import os

from wandbot.ingestion import prepare_dataset, preprocess_data, vectorstores
from wandbot.ingestion.report import create_ingestion_report
from wandbot.utils import get_logger

logger = get_logger(__name__)


def main():
    project = os.environ.get("WANDB_PROJECT", "wandbot-dev")
    entity = os.environ.get("WANDB_ENTITY", "wandbot")

    # Prepare dataset
    raw_artifact = prepare_dataset.load(project, entity)
    # Preprocess dataset
    preprocessed_artifact = preprocess_data.load(project, entity, raw_artifact)
    # Create vectorstore
    vectorstore_artifact = vectorstores.load(project, entity, preprocessed_artifact)
    # TODO: include ingestion report
    create_ingestion_report(
        project, entity, raw_artifact, preprocessed_artifact, vectorstore_artifact
    )


if __name__ == "__main__":
    main()
