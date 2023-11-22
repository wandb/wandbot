import argparse
import os
import pathlib

from wandbot.ingestion import prepare_data, vectorstores
from wandbot.ingestion.report import create_ingestion_report
from wandbot.ingestion.utils import load_custom_dataset_configs_from_yaml
from wandbot.utils import get_logger

logger = get_logger(__name__)

def main(custom: bool, custom_dataset_config_yaml: pathlib.Path):
    project = os.environ.get("WANDB_PROJECT", "wandbot-dev")
    entity = os.environ.get("WANDB_ENTITY", "wandbot")

    if custom and custom_dataset_config_yaml.is_file():
        configs = load_custom_dataset_configs_from_yaml(custom_dataset_config_yaml)
        #TODO: Add the full list of configs as opposed to limiting to one
        #TODO: Add the ability to define which dataloader to use in the config yaml itself
        config = configs[0]
        raw_artifact = prepare_data.load_custom(project, entity, "custom_raw_dataset", config, "docodile")
    else:
        raw_artifact = prepare_data.load(project, entity)
    vectorstore_artifact = vectorstores.load(project, entity, raw_artifact)
    create_ingestion_report(project, entity, raw_artifact, vectorstore_artifact)
    print(vectorstore_artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest data into wandb')
    parser.add_argument('--custom', action='store_true',
                        help='Flag for ingesting a custom dataset')
    parser.add_argument('--custom_dataset_config_yaml', type=pathlib.Path, 
                        default=pathlib.Path(__file__).parent / "custom_dataset.yaml",
                        help='Path to the custom dataset config yaml file')
    args = parser.parse_args()

    main(args.custom, args.custom_dataset_config_yaml)