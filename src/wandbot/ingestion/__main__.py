import os

from wandbot.ingestion import prepare_data, vectorstores
from wandbot.ingestion.report import create_ingestion_report
from wandbot.utils import get_logger

logger = get_logger(__name__)


def main():
    project = os.environ.get("WANDB_PROJECT", "wandbot-dev")
    entity = os.environ.get("WANDB_ENTITY", "wandbot")

    raw_artifact = prepare_data.load(project, entity)
    vectorstore_artifact = vectorstores.load(project, entity, raw_artifact)
    # TODO: include ingestion report
    create_ingestion_report(project, entity, raw_artifact, vectorstore_artifact)
    print(vectorstore_artifact)


if __name__ == "__main__":
    main()
