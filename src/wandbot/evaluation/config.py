from pydantic import Field
from pydantic_settings import BaseSettings


class EvalConfig(BaseSettings):
    evaluation_strategy_name: str = Field(
        "jp v1.2.0-beta",
        description="Will be shown in evaluation page, and be used for just visibility",
    )
    eval_dataset: str = Field(
        "weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA",
        description="Dataset reference for evaluation",
    )
    language: str = Field(
        "ja", description="language for application (en or ja)"
    )

    eval_judge_model: str = Field(
        "gpt-4-1106-preview",
        env="EVAL_JUDGE_MODEL",
        validation_alias="eval_judge_model",
    )
    wandb_entity: str = Field("wandbot", env="WANDB_ENTITY")
    wandb_project: str = Field("wandbot-eval", env="WANDB_PROJECT")
