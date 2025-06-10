import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import requests  # Keep for get_wandbot_configs
import weave
from dotenv import load_dotenv
from weave import Evaluation

# Import new modules
from wandbot.evaluation import data_utils, weave_model, weave_scorer
from wandbot.evaluation.eval_config import EvalConfig, get_eval_config

# EvalChatResponse is now primarily used by weave_model.py, ensure it's correctly located/imported there
# from wandbot.evaluation.eval_schemas import EvalChatResponse
from wandbot.utils import get_logger

# Load environment variables from .env in project root
ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH, override=True)

logger = get_logger(__name__)


def get_wandbot_configs(config: EvalConfig) -> dict:  # config is now required
    """Get wandbot's configs and repo git info"""
    url = config.wandbot_url  # No default, use from config
    try:
        response = requests.get(f"{url}/configs")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error making request to wandbot configs: {e}")
        return {"chat_config": {}, "vector_store_config": {}, "git_info": {}, "app_config": {}}


def main():
    config = get_eval_config()
    logger.info("Starting wandbot evaluation...")
    logger.info(f"Eval Config:\n{vars(config)}\n")

    # Initialize weave with config
    weave.init(f"{config.wandb_entity}/{config.wandb_project}")

    # Use data_utils for loading precomputed data
    precomputed_answers_map = data_utils.load_and_prepare_precomputed_data(config.precomputed_answers_json_path, logger)

    wandbot_info = get_wandbot_configs(config)  # Pass config directly
    if wandbot_info:
        logger.info(f"WandBot configs and git info:\n{wandbot_info}\n")
    else:
        logger.warning("Failed to get WandBot configs")

    os.environ["WEAVE_PARALLELISM"] = str(config.n_weave_parallelism)

    # Use data_utils for loading and preparing dataset rows
    question_rows_for_eval = data_utils.load_and_prepare_dataset_rows(
        config.eval_dataset, config.debug, config.n_debug_samples, logger
    )

    if config.debug:  # This logic is now inside load_and_prepare_dataset_rows, but experiment/eval name changes remain
        config.evaluation_name = f"{config.evaluation_name}_debug"
        config.experiment_name = f"{config.experiment_name}_debug"

    # Initialize Wandbot Model from weave_model.py
    wandbot_model_instance = weave_model.WandbotModel(  # Renamed variable to avoid conflict with module name
        language=config.lang,
        application=config.experiment_name,
        wandbot_url=config.wandbot_url,
        wandbot_config=wandbot_info,
        precomputed_data_map=precomputed_answers_map,
    )

    # Initialize Correctness scorer from weave_scorer.py
    correctness_scorer_instance = weave_scorer.WandbotCorrectnessScorer(config=config)  # Renamed variable

    wandbot_evaluator = Evaluation(
        name=config.evaluation_name,
        dataset=question_rows_for_eval,
        scorers=[correctness_scorer_instance],  # Use instance
        trials=config.n_trials,
    )

    eval_config_summary = {
        "evaluation_strategy_name": config.experiment_name,
        "n_samples": len(question_rows_for_eval),
        "n_trials": config.n_trials,
        "language": config.lang,
        "is_debug": config.debug,
        "eval_judge_model": config.eval_judge_model,
        "eval_judge_temperature": config.eval_judge_temperature,
    }

    eval_attributes = {
        "eval_config": eval_config_summary,
        "wandbot_chat_config": wandbot_info.get("chat_config", {}) if not config.precomputed_answers_json_path else {},
        "wandbot_vectore_store_config": wandbot_info.get("vector_store_config", {})
        if not config.precomputed_answers_json_path
        else {},
        "wandbot_git_info": wandbot_info.get("git_info", {}) if not config.precomputed_answers_json_path else {},
        "wandbot_app_config": wandbot_info.get("app_config", {}),
    }
    # Ensure application name is set correctly even if precomputed answers are used
    if "wandbot_app_config" not in eval_attributes:  # Should always exist based on get_wandbot_configs
        eval_attributes["wandbot_app_config"] = {}
    eval_attributes["wandbot_app_config"]["application"] = config.experiment_name

    logger.info(
        f"Starting evaluation of {len(question_rows_for_eval)} samples with {config.n_trials} trials, \
{len(question_rows_for_eval) * config.n_trials} calls in total."
    )
    with weave.attributes(eval_attributes):
        asyncio.run(
            wandbot_evaluator.evaluate(model=wandbot_model_instance, __weave={"display_name": config.experiment_name})
        )  # Use instance

    all_scored_data = correctness_scorer_instance.get_all_scored_results()  # Use instance
    if all_scored_data:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        output_dir = Path(getattr(config, "eval_output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"evaluation_results_{config.evaluation_name}_{timestamp}.json"
        output_path = output_dir / output_filename

        try:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(all_scored_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved detailed evaluation results to: {output_path.resolve()}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results to {output_path.resolve()}: {e}")
    else:
        logger.warning("No scored results were collected by the scorer to save.")


if __name__ == "__main__":
    main()
