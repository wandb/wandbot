from wandbot.configs.ingestion_config import IngestionConfig
from wandbot.ingestion.create_vector_store import build_vector_store_artifact
from wandbot.ingestion.prepare_data import run_prepare_data_pipeline
from wandbot.ingestion.preprocess_data import run_preprocessing_pipeline
from wandbot.ingestion.report import create_ingestion_report
from wandbot.ingestion.run_ingestion_config import IngestionRunConfig, get_run_config
from wandbot.utils import get_logger

ingestion_config = IngestionConfig()

logger = get_logger(__name__)


def main():
    # Parse command-line arguments
    run_config: IngestionRunConfig = get_run_config()
    logger.info(f"Running ingestion with config: {run_config}")

    project = ingestion_config.wandb_project
    entity = ingestion_config.wandb_entity

    # Adjust artifact names if in debug mode
    raw_data_artifact_name = run_config.raw_data_artifact_name
    preprocessed_data_artifact_name = run_config.preprocessed_data_artifact_name
    vectorstore_artifact_name = run_config.vectorstore_artifact_name

    if run_config.debug:
        logger.warning("----- RUNNING IN DEBUG MODE -----")
        raw_data_artifact_name += "_debug"
        preprocessed_data_artifact_name += "_debug"
        vectorstore_artifact_name += "_debug"
        logger.info(f"Debug mode: Artifact names adjusted to: {raw_data_artifact_name}, {preprocessed_data_artifact_name}, {vectorstore_artifact_name}")

    # Variables to hold artifact paths/names
    raw_artifact_path = None
    preprocessed_artifact_path = None
    vectorstore_artifact_path = None

    # Execute steps based on config
    if "prepare" in run_config.steps:
        logger.info("\n\n ------ Starting Prepare Data Step ------\n\n")
        raw_artifact_path = run_prepare_data_pipeline(
            project=project,
            entity=entity,
            result_artifact_name=raw_data_artifact_name,
            include_sources=run_config.include_sources,
            exclude_sources=run_config.exclude_sources,
            debug=run_config.debug,
        )
        logger.info(f"Prepare Data Step completed. Raw artifact: {raw_artifact_path}")
    else:
        logger.info("Skipping Prepare Data Step")

    if "preprocess" in run_config.steps:
        logger.info("\n\n ------ Starting Preprocess Data Step ------\n\n")
        if not raw_artifact_path:
            raw_artifact_path = (
                f"{entity}/{project}/{raw_data_artifact_name}:latest"
            )
            logger.warning(
                f"Prepare step skipped, using latest raw artifact: {raw_artifact_path}"
            )

        preprocessed_artifact_path = run_preprocessing_pipeline(
            project=project,
            entity=entity,
            source_artifact_path=raw_artifact_path,
            result_artifact_name=preprocessed_data_artifact_name,
            debug=run_config.debug,
        )
        logger.info(
            f"Preprocess Data Step completed. Preprocessed artifact: {preprocessed_artifact_path}"
        )
    else:
        logger.info("Skipping Preprocess Data Step")

    if "vectorstore" in run_config.steps:
        logger.info("\n\n ------ Starting Vector Store Step ------\n\n")
        if not preprocessed_artifact_path:
            preprocessed_artifact_path = f"{entity}/{project}/{preprocessed_data_artifact_name}:latest"
            logger.warning(
                f"Preprocess step skipped, using latest preprocessed artifact: {preprocessed_artifact_path}"
            )

        vectorstore_artifact_path = build_vector_store_artifact(
            project=project,
            entity=entity,
            source_artifact_path=preprocessed_artifact_path,
            result_artifact_name=vectorstore_artifact_name,
            debug=run_config.debug,
        )
        logger.info(
            f"Vector Store Step completed. Vectorstore artifact: {vectorstore_artifact_path}"
        )
    else:
        logger.info("Skipping Vector Store Step")

    if "report" in run_config.steps:
        logger.info("\n\n ------ Starting Report Creation Step ------\n\n")
        if not raw_artifact_path:
            raw_artifact_path = (
                f"{entity}/{project}/{raw_data_artifact_name}:latest"
            )
            logger.warning(
                f"Prepare step skipped, using latest raw artifact for report: {raw_artifact_path}"
            )
        if not vectorstore_artifact_path:
            vectorstore_artifact_path = (
                f"{entity}/{project}/{vectorstore_artifact_name}:latest"
            )
            logger.warning(
                f"Vectorstore step skipped, using latest vectorstore artifact for report: {vectorstore_artifact_path}"
            )

        create_ingestion_report(
            project, entity, raw_artifact_path, vectorstore_artifact_path, debug=run_config.debug
        )
        logger.info("Report Step completed.")
    else:
        logger.info("Skipping Report Step")

    logger.info("Ingestion pipeline finished.")
    final_artifact = (
        vectorstore_artifact_path
        or preprocessed_artifact_path
        or raw_artifact_path
        or "No artifact generated."
    )
    print(f"Final artifact from run: {final_artifact}")


if __name__ == "__main__":
    main()
