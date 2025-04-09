from wandbot.configs.ingestion_config import IngestionConfig
from wandbot.ingestion.prepare_data import run_prepare_data_pipeline
from wandbot.ingestion.preprocess_data import run_preprocessing_pipeline
from wandbot.ingestion.run_ingestion_config import IngestionRunConfig, get_run_config
from wandbot.ingestion.vectorstore_and_report import run_vectorstore_and_report_pipeline
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

    # Combine Vectorstore and Report steps if either is requested
    if "vectorstore" in run_config.steps or "report" in run_config.steps:
        logger.info("\n\n ------ Starting Combined Vector Store and Report Step ------\n\n")

        # Ensure raw_artifact_path is set (needed for report)
        if not raw_artifact_path:
            raw_artifact_path = f"{entity}/{project}/{raw_data_artifact_name}:latest"
            logger.warning(
                f"Prepare step skipped, using latest raw artifact for vectorstore/report: {raw_artifact_path}"
            )

        # Ensure preprocessed_artifact_path is set (needed for vectorstore)
        if not preprocessed_artifact_path:
            preprocessed_artifact_path = f"{entity}/{project}/{preprocessed_data_artifact_name}:latest"
            logger.warning(
                f"Preprocess step skipped, using latest preprocessed artifact for vectorstore/report: {preprocessed_artifact_path}"
            )

        create_report_flag = "report" in run_config.steps
        if not create_report_flag:
             logger.info("Report creation will be skipped as 'report' is not in the specified steps.")

        vectorstore_artifact_path = run_vectorstore_and_report_pipeline(
            project=project,
            entity=entity,
            raw_artifact_path=raw_artifact_path,
            preprocessed_artifact_path=preprocessed_artifact_path,
            vectorstore_artifact_name=vectorstore_artifact_name,
            debug=run_config.debug,
            create_report=create_report_flag
        )
        logger.info(
            f"Combined Vector Store and Report Step completed. Vectorstore artifact: {vectorstore_artifact_path}"
        )

    else:
        logger.info("Skipping Combined Vector Store and Report Step")

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