from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalConfig(BaseSettings):
    evaluation_strategy_name: str = Field("ja v1.2.0-beta gpt3.5", description="Will be shown in evaluation page, and be used for just visibility")
    eval_dataset: str = Field(
        "weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA"
        ,description="Dataset reference for evaluation"
    )
    # en evaluation dataset: "weave:///wandbot/wandbot-eval/object/wandbot_eval_data:eCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU"
    # jp evaluation dataset: "weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA" 
    # jp small evaluation dataset: "weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp_test:Qp5qAlNYhzLJSfDZONaBNDzjhEmlfOTXNI1NvYhELKQ"

    language: str = Field("ja", description="language for application (en or ja)")

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
    eval_judge_model: str = Field(
        "gpt-4-1106-preview",
        env="EVAL_JUDGE_MODEL",
        validation_alias="eval_judge_model",
    )
    wandb_entity: str = Field("wandbot", env="WANDB_ENTITY")
    wandb_project: str = Field("wandbot-eval", env="WANDB_PROJECT")

    
