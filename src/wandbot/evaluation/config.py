from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    eval_artifact: str = Field(
        "wandbot/wandbot-eval/autoeval_dataset:v3",
        env="EVAL_ARTIFACT",
        validation_alias="eval_artifact",
    )
    eval_artifact_root: str = Field(
        "data/eval",
        env="EVAL_ARTIFACT_ROOT",
        validation_alias="eval_artifact_root",
    )

    eval_annotations_file: str = Field(
        "wandbot_cleaned_annotated_dataset_11-12-2023.jsonl",
        env="EVAL_ANNOTATIONS_FILE",
        validation_alias="eval_annotations_file",
    )
    eval_output_file: str = Field(
        "eval.jsonl",
        env="EVAL_OUTPUT_FILE",
        validation_alias="eval_output_file",
    )
    wandb_entity: str = Field("wandbot", env="WANDB_ENTITY")
    wandb_project: str = Field("wandbot-eval", env="WANDB_PROJECT")
