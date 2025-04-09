from wandbot.configs.ingestion_config import IngestionConfig
from wandbot.ingestion.prepare_data import run_prepare_data_pipeline
from wandbot.ingestion.preprocess_data import run_preprocessing_pipeline
from wandbot.ingestion.vectorstore_and_report import run_vectorstore_and_report_pipeline
from wandbot.ingestion.run_ingestion_config import IngestionRunConfig, get_run_config
from wandbot.utils import get_logger
import wandb
from typing import Optional, Any

ingestion_config = IngestionConfig()

logger = get_logger(__name__)

def get_artifact_base_name(artifact_full_name: str) -> str:
    """Extracts the base name (without version/alias) from an artifact path."""
    return artifact_full_name.split("/")[-1].split(":")[0]

def main():
    run_config: IngestionRunConfig = get_run_config()
    logger.info(f"Running ingestion with config: {run_config}")

    project = ingestion_config.wandb_project
    entity = ingestion_config.wandb_entity

    raw_data_artifact_name = run_config.raw_data_artifact_name
    preprocessed_data_artifact_name = run_config.preprocessed_data_artifact_name
    vectorstore_artifact_name = run_config.vectorstore_artifact_name

    if run_config.debug:
        logger.warning("----- RUNNING IN DEBUG MODE ----- ")
        raw_data_artifact_name += "_debug"
        preprocessed_data_artifact_name += "_debug"
        vectorstore_artifact_name += "_debug"
        logger.info(f"Debug mode: Artifact names adjusted to: {raw_data_artifact_name}, {preprocessed_data_artifact_name}, {vectorstore_artifact_name}")

    raw_artifact_path: Optional[str] = None
    preprocessed_artifact_path: Optional[str] = None
    vectorstore_artifact_path: Optional[str] = None
    final_artifact_to_print: Optional[str] = None

    # Determine if the combined step should run
    run_combined_vectorstore_report = "vectorstore_report" in run_config.steps

    try:
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
            final_artifact_to_print = raw_artifact_path
        else:
            logger.info("Skipping Prepare Data Step")
            # Set path to latest if needed later
            raw_artifact_path = f"{entity}/{project}/{raw_data_artifact_name}:latest"

        if "preprocess" in run_config.steps:
            logger.info("\n\n ------ Starting Preprocess Data Step ------\n\n")
            # raw_artifact_path is guaranteed to be set here (either from prepare or defaulted above)
            preprocessed_artifact_path = run_preprocessing_pipeline(
                project=project,
                entity=entity,
                source_artifact_path=raw_artifact_path, # Safe to use
                result_artifact_name=preprocessed_data_artifact_name,
                debug=run_config.debug,
            )
            logger.info(
                f"Preprocess Data Step completed. Preprocessed artifact: {preprocessed_artifact_path}"
            )
            final_artifact_to_print = preprocessed_artifact_path
        else:
            logger.info("Skipping Preprocess Data Step")
            # Set path to latest if needed later
            preprocessed_artifact_path = f"{entity}/{project}/{preprocessed_data_artifact_name}:latest"

        # --- Combined Vector Store and Report Step ---
        if run_combined_vectorstore_report:
            logger.info("\n\n ------ Starting Combined Vector Store and Report Step ------\n\n")
            # Ensure previous steps artifacts are available
            if not raw_artifact_path:
                raw_artifact_path = f"{entity}/{project}/{raw_data_artifact_name}:latest"
                logger.warning(f"Prepare step skipped, using latest raw artifact for combined step: {raw_artifact_path}")
            if not preprocessed_artifact_path:
                preprocessed_artifact_path = f"{entity}/{project}/{preprocessed_data_artifact_name}:latest"
                logger.warning(f"Preprocess step skipped, using latest preprocessed artifact for combined step: {preprocessed_artifact_path}")
            
            vectorstore_artifact_path = run_vectorstore_and_report_pipeline(
                project=project,
                entity=entity,
                raw_artifact_path=raw_artifact_path,             # Pass raw path
                preprocessed_artifact_path=preprocessed_artifact_path, # Pass preprocessed path
                vectorstore_artifact_name=vectorstore_artifact_name, # Pass VS artifact name
                debug=run_config.debug,
                create_report=True, # Explicitly create report when run via CLI step
            )
            logger.info(
                f"Combined Vector Store and Report Step completed. Vectorstore artifact: {vectorstore_artifact_path}"
            )
            final_artifact_to_print = vectorstore_artifact_path
        
        # Simplified logic: if the combined step wasn't run, log skip message
        elif not run_combined_vectorstore_report:
             logger.info("Skipping Vector Store and Report Step as 'vectorstore_report' not specified.")
             # Set path to latest if subsequent steps might need it (though none currently do)
             vectorstore_artifact_path = f"{entity}/{project}/{vectorstore_artifact_name}:latest"

        logger.info("Ingestion pipeline finished.")
        print(f"Final artifact from run: {final_artifact_to_print or 'No artifact generated by executed steps.'}")

    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)
        # Combined step handles its own run finishing; standalone report handles its own finishing.
        # Need to consider if any other run could be lingering. The original VS step now finishes its own run.
        raise # Re-raise the exception
    # No final 'finally: run.finish()' needed here as each step/combined step manages its own run.

if __name__ == "__main__":
    main()
