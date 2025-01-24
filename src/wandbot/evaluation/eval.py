import os
import json
import httpx
import weave
import asyncio
import re
import logging
import requests
from weave import Evaluation
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, after_log

from wandbot.utils import get_logger
from wandbot.evaluation.eval_metrics.correctness import WandBotCorrectnessEvaluator
from wandbot.evaluation.eval_config import get_eval_config, EvalConfig
from dotenv import load_dotenv

dot_env_path = os.path.join(os.path.dirname(__file__), '../../../../.env')
load_dotenv(dotenv_path=dot_env_path, override=True)

logger = get_logger(__name__)

def get_wandbot_configs(config: EvalConfig = None):
    """Get wandbot's configs and repo git info"""
    url = config.wandbot_url if config else "http://0.0.0.0:8000"
    try:
        response = requests.get(f"{url}/configs")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error making request to wandbot configs: {e}")
        return {
            "chat_config": {},
            "vector_store_config": {},
            "git_info": {},
            "app_config": {}
        }

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=10, max=300),
    retry=retry_if_exception_type(httpx.HTTPError),
    before_sleep=lambda retry_state: logger.warning(
        f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
    ),
    after=after_log(logger, logging.ERROR)
)
async def make_request(url: str, question: str, application: str = "api-eval", language: str = "en") -> dict:
    """Make HTTP request to wandbot API with retry logic."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=900.0, connect=30.0)) as client:
        try:
            response = await client.post(
                f"{url}/chat/query",
                json={"question": question, "application": application, "language": language}
            )
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            logger.error("Request timed out after 900 seconds")
            raise
        except httpx.ConnectTimeout:
            logger.error("Connection timed out after 30 seconds")
            raise

async def get_answer(question: str, wandbot_url: str, application: str = "api-eval", language: str = "en") -> str:
    """Get answer from wandbot API."""
    try:
        result = await make_request(wandbot_url, question, application, language)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Failed to get answer: {str(e)}")
        return json.dumps({
            "error": str(e),
            "answer": "",
            "system_prompt": "",
            "source_documents": "",
            "model": "",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "time_taken": 0
        })

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
async def get_record(question: str, wandbot_url: str, application: str = "api-eval", language: str = "en") -> dict:
    try:
        response = await get_answer(question, wandbot_url=wandbot_url, application=application, language=language)
        response_dict = json.loads(response)
        
        if not response_dict:
            try:
                error_data = json.loads(response)
                error_msg = error_data.get("error", "Unknown API error")
            except json.JSONDecodeError:
                error_msg = response if response else "Empty response from API"
            
            logger.error(error_msg)
            return {
                "system_prompt": "",
                "generated_answer": "",
                "retrieved_contexts": [],
                "model": "",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "time_taken": 0,
                "has_error": True,
                "error_message": error_msg
            }
        
        return {
            "system_prompt": response_dict.get("system_prompt", ""),
            "generated_answer": response_dict.get("answer", ""),
            "retrieved_contexts": parse_text_to_json(
                response_dict.get("source_documents", "")
            ),
            "model": response_dict.get("model", ""),
            "total_tokens": response_dict.get("total_tokens", 0),
            "prompt_tokens": response_dict.get("prompt_tokens", 0),
            "completion_tokens": response_dict.get("completion_tokens", 0),
            "time_taken": response_dict.get("time_taken", 0),
            "has_error": False,
            "error_message": None
        }
    except Exception as e:
        error_msg = f"Error getting answer from API: {str(e)}"
        logger.error(error_msg)
        return {
            "system_prompt": "",
            "generated_answer": "",
            "retrieved_contexts": [],
            "model": "",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "time_taken": 0,
            "has_error": True,
            "error_message": error_msg
        }


class WandbotModel(weave.Model):
    language: str = "en"
    application: str = "api-eval"
    wandbot_url: str = "http://0.0.0.0:8000"

    @weave.op
    async def predict(self, question: str) -> dict:
        prediction = await get_record(question, 
                                      wandbot_url=self.wandbot_url, 
                                      application=self.application,
                                      language=self.language)
        return prediction


class WandbotCorrectnessScorer(weave.Scorer):
    config: EvalConfig
    correctness_evaluator: WandBotCorrectnessEvaluator = None
    debug: bool = False

    def __init__(self, config: EvalConfig):
        super().__init__(config=config)
        self.debug = config.debug
        self.correctness_evaluator = WandBotCorrectnessEvaluator(
            client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=config.max_evaluator_retries, timeout=config.evaluator_timeout),
            model=config.eval_judge_model,
            temperature=config.eval_judge_temperature,
        )
    
    @weave.op
    async def score(self, question: str, ground_truth: str, notes: str, model_output: dict) -> dict:

        if self.debug:
            if model_output is not None:
                logger.info(f"In get_answer_correctness, model_output keys:\n{model_output.keys()}")
            else:
                logger.error("model_output is None")
    
        try:
            contexts = [c["content"] for c in model_output.get("retrieved_contexts", [])] if model_output.get("retrieved_contexts") else []
            
            if model_output.get("generated_answer", "") == "":
                error_msg = "Generated answer is empty"
                logger.error(error_msg)
                return {
                    "answer_correct": False,
                    "reasoning": error_msg,
                    "score": 1.0,
                    "has_error": True,
                    "error_message": error_msg
                }
            if model_output.get("has_error", False):
                error_msg = model_output.get("error_message", "Unknown error")
                logger.error(error_msg)
                return {
                    "answer_correct": False,
                    "reasoning": error_msg,
                    "score": 1.0,
                    "has_error": True,
                    "error_message": error_msg
                }
            
            # If not error from wandbot generation, run the correctness evaluator
            result = await self.correctness_evaluator.aevaluate(
                query=question,
                response=model_output.get("generated_answer", ""),
                reference=ground_truth,
                contexts=contexts,
                reference_notes=notes,
            )
            res = result.model_dump()
            return {
                "answer_correct": res.get("passing", None),
                "reasoning": res.get("reasoning", ""),
                "score": res.get("score", None),
                "has_error": False,
                "error_message": ""
            }
        except Exception as e:
            error_msg = f"Error evaluating answer: {str(e)}"
            logger.error(error_msg)
            return {
                "answer_correct": False,
                "reasoning": "Evaluation failed due to an error",
                "score": 1.0,
                "has_error": True,
                "error_message": error_msg
            }


def main():
    config = get_eval_config()
    logger.info("Starting wandbot evaluation...")
    logger.info(f"Eval Config:\n{vars(config)}\n")

    # Initialize weave with config
    weave.init(f"{config.wandb_entity}/{config.wandb_project}")

    wandbot_info = get_wandbot_configs(config)
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

    # Initialize Wandbot Model
    wandbot = WandbotModel(language=config.lang, 
                           application=config.experiment_name, 
                           wandbot_url=config.wandbot_url)
    
    # Initialize Correctness scorer
    correctness_scorer = WandbotCorrectnessScorer(config=config)

    wandbot_evaluator = Evaluation(
        name=config.evaluation_name,
        dataset=question_rows, 
        scorers=[correctness_scorer],
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
            "wandbot_chat_config": wandbot_info.get("chat_config", {}),
            "wandbot_vectore_store_config": wandbot_info.get("vector_store_config", {}),
            "wandbot_git_info": wandbot_info.get("git_info", {}),
            "wandbot_app_config": wandbot_info.get("app_config", {})
            }

    logger.info("Number of evaluation samples: %s", len(question_rows))
    logger.info(f"Starting evaluation of {len(question_rows)} samples with {config.n_trials} trials")
    with weave.attributes(eval_attributes):
        asyncio.run(wandbot_evaluator.evaluate(
            model=wandbot, __weave={"display_name": config.experiment_name}
        ))

if __name__ == "__main__":
    main()