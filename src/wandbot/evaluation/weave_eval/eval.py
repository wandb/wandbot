import os
import json
import httpx
import weave
import asyncio
import re
import logging
import requests
from weave import Evaluation
from llama_index.llms.openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from wandbot.utils import get_logger
from wandbot.evaluation.eval.correctness import (
    CORRECTNESS_EVAL_TEMPLATE,
    WandBotCorrectnessEvaluator,
)
from wandbot.evaluation.eval_config import get_eval_config
from tenacity import wait_random, after_log

from dotenv import load_dotenv

dot_env_path = os.path.join(os.path.dirname(__file__), '../../../../.env')
load_dotenv(dotenv_path=dot_env_path, override=True)

logger = get_logger(__name__)

config = get_eval_config()
WANDBOT_URL = config.wandbot_url or "http://0.0.0.0:8000"

weave.init(f"{config.wandb_entity}/{config.wandb_project}")

wandbot_correctness_evaluator = WandBotCorrectnessEvaluator(
    llm=OpenAI(model=config.eval_judge_model, temperature=config.eval_judge_temperature),
    eval_template=CORRECTNESS_EVAL_TEMPLATE,
)

    
def get_wandbot_configs():
    """Get wandbot's configs and repo git info"""
    try:
        response = requests.get('http://0.0.0.0:8000/configs')
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error making request: {e}")
        return None
    

@weave.op
async def get_answer(question: str, application: str = "api-eval", language: str = "en") -> str:
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=600) + wait_random(0, 2),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.RequestError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
        ),
        after=after_log(logger, logging.ERROR)
    )
    async def _make_request():
        async with httpx.AsyncClient(timeout=900.0) as client:
            response = await client.post(
                f"{WANDBOT_URL}/chat/query",
                json={"question": question, "application": application, "language": language}
            )
            response.raise_for_status()
            return response.json()
    try:
        result = await _make_request()
        return json.dumps(result)
    except Exception as e:
        logger.error(f"All retry attempts failed, returing an empty dict. Error: {str(e)}")
        return json.dumps({})


def parse_text_to_json(text):
    # Split the text into documents
    documents = re.split(r"source: https?://", text)[1:]
    result = []
    for doc in documents:
        source_url = "https://" + doc.split("\n")[0].strip()
        content = "\n".join(doc.split("\n")[1:]).strip()
        document = {"source": source_url, "content": content}
        result.append(document)
    return result


@weave.op
async def get_record(question: str, application: str = "api-eval", language: str = "en") -> dict:
    response = await get_answer(question, application, language=language)
    response = json.loads(response)
    
    # Return default values if response is empty or missing fields
    if not response:
        return {
            "system_prompt": "",
            "generated_answer": "",
            "retrieved_contexts": [],
            "model": "",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "time_taken": 0,
        }
    
    return {
        "system_prompt": response.get("system_prompt", ""),
        "generated_answer": response.get("answer", ""),
        "retrieved_contexts": parse_text_to_json(
            response.get("source_documents", "")
        ),
        "model": response.get("model", ""),
        "total_tokens": response.get("total_tokens", 0),
        "prompt_tokens": response.get("prompt_tokens", 0),
        "completion_tokens": response.get("completion_tokens", 0),
        "time_taken": response.get("time_taken", 0),
    }


class WandbotModel(weave.Model):
    language: str = "en"
    application: str = "api-eval"

    @weave.op
    async def predict(self, question: str) -> dict:
        prediction = await get_record(question, application=self.application, language=self.language)
        return prediction

@weave.op
async def answer_correctness_scorer(
    question: str, ground_truth: str, notes: str, model_output: dict
) -> dict:
    if config.debug:
        if model_output is not None:
            logger.info(f"In get_answer_correctness, model_output keys:\n{model_output.keys()}")
        else:
            logger.error("model_output is None")
    contexts = [c["content"] for c in model_output.get("retrieved_contexts", [])] if model_output.get("retrieved_contexts") else []
    result = await wandbot_correctness_evaluator.aevaluate(
        query=question,
        response=model_output["generated_answer"],
        reference=ground_truth,
        contexts=contexts,
        reference_notes=notes,
    )
    res = result.dict()
    return {
        "answer_correct": res.get("passing", None),
        "reasoning": res.get("feedback"),
        "score": res.get("score"),
        "invalid_result": res.get("invalid_result"),
        "invalid_reasoning": res.get("invalid_result"),
    }


def main():
    logger.info("Starting wandbot evaluation...")
    logger.info(f"Eval Config:\n{vars(config)}\n")

    wandbot_info = get_wandbot_configs()
    if wandbot_info: 
        logger.info(f"WandBot configs and git info:\n{wandbot_info}\n")
    else:
        logger.warning("Failed to get WandBot configs")

    os.environ["WEAVE_PARALLELISM"] = str(config.n_weave_parallelism)

    dataset_ref = weave.ref(config.eval_dataset).get()
    question_rows = dataset_ref.rows

    if config.debug:
        question_rows = question_rows[:config.n_debug_samples]
        config.evaluation_name = f"{config.evaluation_name}_debug"
        config.experiment_name = f"{config.experiment_name}_debug"

    question_rows = [
        {
            "question": row["question"],
            "ground_truth": row["answer"],
            "notes": row["notes"],
        }
        for row in question_rows
    ]
    logger.info("Number of evaluation samples: %s", len(question_rows))

    wandbot = WandbotModel(language=config.lang, application=config.experiment_name)

    wandbot_evaluator = Evaluation(
        name=config.evaluation_name,
        dataset=question_rows, 
        scorers=[answer_correctness_scorer],
        trials=config.n_trials
    )

    eval_config = {
            "evaluation_strategy_name": config.experiment_name,
            "n_samples": len(question_rows),
            "n_trials": config.n_trials,
            "language": config.lang,
            "is_debug": config.debug,
            "eval_judge_model": config.eval_judge_model,
            "eval_judge_temperature": config.eval_judge_temperature,
    }

    eval_attributes = {
            "eval_config": eval_config,
            "wandbot_chat_config": (wandbot_info["chat_config"] or {}),
            "wandbot_vectore_store_config": (wandbot_info["vector_store_config"] or {}),
            "wandbot_git_info": (wandbot_info["git_info"] or {}),
            "wandbot_app_config": (wandbot_info["app_config"] or {})
            }

    logger.info(f"Starting evaluation of {len(question_rows)} samples with {config.n_trials} trials")
    with weave.attributes(eval_attributes):
        asyncio.run(wandbot_evaluator.evaluate(
            model=wandbot, __weave={"display_name": config.experiment_name}
        ))

if __name__ == "__main__":
    main()