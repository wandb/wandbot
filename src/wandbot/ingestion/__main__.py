import argparse
import os

from wandbot.ingestion import prepare_data, preprocess_data, vectorstores
from wandbot.ingestion.report import create_ingestion_report
from wandbot.utils import get_logger, load_config

logger = get_logger(__name__)


def main(config_path: str) -> None:
    config = load_config(config_path)
    logger.info(config)

    project = os.environ.get("WANDB_PROJECT", "wandbot-dev")
    entity = os.environ.get("WANDB_ENTITY", "wandbot")

    # raw_artifact = prepare_data.load(project, entity)
    raw_artifact = "wandbot/wandbot-dev/raw_dataset:v55"
    logger.info(f"Loaded all the data sources at {raw_artifact}")

    preprocessed_artifact = preprocess_data.load(
        project, entity, raw_artifact, config
    )
    # preprocessed_artifact = "wandbot/wandbot-dev/transformed_data:latest"
    logger.info(
        f"Data sources preprocessed and stored at {preprocessed_artifact}"
    )

    vectorstore_artifact = vectorstores.load(
        project, entity, preprocessed_artifact, config
    )
    logger.info(
        f"Preprocessed chunks embedded and stored at {vectorstore_artifact}"
    )

    create_ingestion_report(project, entity, raw_artifact, vectorstore_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ingestion process with a specified configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )

    args = parser.parse_args()
    main(args.config)
