from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalConfig(BaseSettings):
    evaluation_strategy_name: str = Field("en v1.2.0", description="Will be shown in evaluation page, and be used for just visibility")
    eval_dataset: str = Field(
        "weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA"
        ,description="Dataset reference for evaluation"
    )
    language: str = Field("en", description="language for application (en or ja or kr)")

    eval_judge_model: str = Field(
        "gpt-4-1106-preview",
        env="EVAL_JUDGE_MODEL",
        validation_alias="eval_judge_model",
    )
    wandb_entity: str = Field("wandbot", env="WANDB_ENTITY")
    wandb_project: str = Field("wandbot-eval-kr", env="WANDB_PROJECT")

    #en: weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA
    #jp: weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA
    #kr: weave:///wandbot/wandbot-eval-kr/object/wandbot_eval_data_kr:9vnf9CU5V8eKkmHoiACyB01oQwtY3RmIo0XGH2cVhvo

    
