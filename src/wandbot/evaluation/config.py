from dataclasses import dataclass
import simple_parsing as sp

@dataclass
class EvalConfig:
    lang: str = "en"  # language for eval dataset to use (en or ja)
    eval_judge_model: str = "gpt-4-1106-preview"
    eval_judge_temperature: float = 0.1
    experiment_name: str = "wandbot-eval"
    evaluation_name: str = "wandbot-eval"
    n_trials: int = 3
    n_weave_parallelism: int = 20
    wandb_entity: str = "wandbot"
    wandb_project: str = "wandbot-eval"
    debug: bool = False
    n_debug_samples: int = 3

    @property
    def eval_dataset(self) -> str:
        if self.lang == "ja":
            return "weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA"
        return "weave:///wandbot/wandbot-eval/object/wandbot_eval_data:eCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU"


def get_config() -> EvalConfig:
    return sp.parse(EvalConfig)
