import os

from wandbot.ingestion import preprocess_data, vectorstores
from wandbot.utils import get_logger

logger = get_logger(__name__)


def main():
    project = os.environ.get("WANDB_PROJECT", "wandbot-dev")
    entity = os.environ.get("WANDB_ENTITY", "wandbot")

    # raw_artifact = prepare_data.load(project, entity)
    raw_artifact = "wandbot/wandbot-dev/raw_dataset:v39"
    preprocessed_artifact = preprocess_data.load(project, entity, raw_artifact)
    # preprocessed_artifact = "wandbot/wandbot-dev/transformed_data:latest"
    vectorstore_artifact = vectorstores.load(
        project, entity, preprocessed_artifact
    )
    # TODO: include ingestion report
    # create_ingestion_report(project, entity, raw_artifact, vectorstore_artifact)
    print(vectorstore_artifact)


if __name__ == "__main__":
    main()
