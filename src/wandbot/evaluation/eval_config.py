from dataclasses import dataclass
from typing import Literal

import simple_parsing as sp


@dataclass
class EvalConfig:
    # language for eval dataset to use (en or ja)
    lang: Literal["en", "ja"] = "en"
    eval_judge_provider: Literal["anthropic", "openai"] = "openai"
    eval_judge_model: str = "gpt-4o-2024-11-20"
    eval_judge_temperature: float = 0.1
    experiment_name: str = "wandbot-eval"
    evaluation_name: str = "wandbot-eval"
    n_trials: int = 3
    n_weave_parallelism: int = 10
    wandbot_url: str = "http://0.0.0.0:8000"
    wandb_entity: str = "wandbot"
    wandb_project: str = "wandbot-eval"
    debug: bool = False
    n_debug_samples: int = 3
    max_evaluator_retries: int = 3
    evaluator_timeout: int = 60
    precomputed_answers_json_path: str | None = sp.field(default=None, help="Path to a JSON file containing precomputed answers. If provided, network calls to wandbot will be skipped.")

    # Links to evaluation datasets stored in Weave
    @property
    def eval_dataset(self) -> str:
        if self.lang == "ja":
            return "weave:///wandbot/wandbot-eval/object/wandbot_eval_data_jp:I2BlFnw1VnPn8lFG72obBWN1sCokB3EYk4G4vSKg23g"
        return "weave:///wandbot/wandbot-eval/object/wandbot_eval_data:ZZUQa2CCAqPDFWiB90VANCm4EdT8qtc125NazaWUrdI"

def get_eval_config() -> EvalConfig:
    return sp.parse(EvalConfig)
