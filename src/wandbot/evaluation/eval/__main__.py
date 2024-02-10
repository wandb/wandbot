import json
from typing import Any, Hashable

import nest_asyncio
import pandas as pd
import requests
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from ragas import metrics
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from wandbot.evaluation.eval.correctness import (
    CORRECTNESS_EVAL_TEMPLATE,
    WandbCorrectnessEvaluator,
)
from wandbot.evaluation.eval.factfulness import (
    FACTFULNESS_EVAL_TEMPLATE,
    WandbFactfulnessEvaluator,
)
from wandbot.evaluation.eval.relevancy import (
    RELEVANCY_EVAL_TEMPLATE,
    WandbRelevancyEvaluator,
)
from wandbot.utils import cachew, get_logger

logger = get_logger(__name__)

nest_asyncio.apply()


EVAL_CACHE = "data/cache/eval_cache/cache.db"
service_context = ServiceContext.from_defaults(llm=OpenAI("gpt-4-1106-preview"))
correctness_evaluator = WandbCorrectnessEvaluator(
    service_context=service_context,
    eval_template=CORRECTNESS_EVAL_TEMPLATE,
)
faithfulness_evaluator = WandbFactfulnessEvaluator(
    service_context=service_context,
    eval_template=FACTFULNESS_EVAL_TEMPLATE,
)
relevancy_evaluator = WandbRelevancyEvaluator(
    service_context=service_context,
    eval_template=RELEVANCY_EVAL_TEMPLATE,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@cachew(cache_path=EVAL_CACHE, logger=logger)
def get_answer(question: str, application: str = "api-eval-bharat") -> str:
    url = "http://0.0.0.0:8000/query"
    payload = {
        "question": question,
        "language": "en",
        "application": application,
    }
    response = requests.post(url, data=json.dumps(payload))
    response = response.json()
    return json.dumps(response)


@cachew(cache_path=EVAL_CACHE, logger=logger)
def get_eval_record(row_str: str, application: str = "api-eval-bharat") -> str:
    row = json.loads(row_str)
    response = get_answer(row["question"], application=application)
    response = json.loads(response)
    response["ground_truths"] = row["answer"]
    response["reference_notes"] = row["notes"]
    response["contexts"] = [
        "\nSource: " + source["source"] + " \n " + source["text"]
        for source in json.loads(response["source_documents"])
    ]
    response = json.dumps(response)
    return response


def parse_answer_eval(metric: str, row: dict[str, Any]) -> dict[str, Any]:
    result = {
        f"{metric}_score": row.get("score"),
        f"{metric}_result": row.get("passing"),
        f"{metric}_reason": row.get("feedback"),
    }
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@cachew(cache_path=EVAL_CACHE, logger=logger)
def get_answer_correctness(row_str: str) -> str:
    row = json.loads(row_str)
    result = correctness_evaluator.evaluate(
        query=row["question"],
        response=row["answer"],
        reference=row["ground_truths"],
        contexts=row["contexts"],
        reference_notes=row["reference_notes"],
    )
    result = parse_answer_eval("answer_correctness", result.dict())
    result[
        "answer_correctness_score_(ragas)"
    ] = metrics.answer_correctness.score_single(row)
    result = json.dumps(result)
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@cachew(cache_path=EVAL_CACHE, logger=logger)
def get_answer_relevancy(row_str: str) -> str:
    row = json.loads(row_str)
    result = relevancy_evaluator.evaluate(
        query=row["question"],
        response=row["answer"],
        contexts=row["contexts"],
        reference=row["ground_truths"],
    )
    result = parse_answer_eval("answer_relevancy", result.dict())
    result[
        "answer_relevancy_score_(ragas)"
    ] = metrics.answer_relevancy.score_single(row)
    result = json.dumps(result)
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@cachew(cache_path=EVAL_CACHE, logger=logger)
def get_answer_faithfulness(row_str: str) -> str:
    row = json.loads(row_str)
    result = faithfulness_evaluator.evaluate(
        query=row["question"],
        response=row["answer"],
        contexts=row["contexts"],
        reference=row["ground_truths"],
    )

    result = parse_answer_eval("answer_faithfulness", result.dict())
    result[
        "answer_faithfulness_score_(ragas)"
    ] = metrics.faithfulness.score_single(row)
    result = json.dumps(result)
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@cachew(cache_path=EVAL_CACHE, logger=logger)
def get_answer_similarity(row_str: str) -> str:
    row = json.loads(row_str)
    result = metrics.answer_similarity.score_single(row)
    result = json.dumps({"answer_similarity_score_(ragas)": result})
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@cachew(cache_path=EVAL_CACHE, logger=logger)
def get_context_precision(row_str: str) -> str:
    row = json.loads(row_str)
    result = metrics.context_precision.score_single(row)
    result = json.dumps({"context_precision_score": result})
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@cachew(cache_path=EVAL_CACHE, logger=logger)
def get_context_recall(row_str: str) -> str:
    row = json.loads(row_str)
    result = metrics.context_recall.score_single(row)
    result = json.dumps({"context_recall_score": result})
    return result


@cachew(cache_path=EVAL_CACHE, logger=logger)
def evaluate_row(idx: Hashable, row_str: str) -> str:
    eval_result = {"idx": idx}
    row = json.loads(row_str)
    eval_result.update(row)
    eval_result.update(json.loads(get_answer_correctness(row_str)))
    eval_result.update(json.loads(get_answer_relevancy(row_str)))
    eval_result.update(json.loads(get_answer_faithfulness(row_str)))
    eval_result.update(json.loads(get_answer_similarity(row_str)))
    eval_result.update(json.loads(get_context_precision(row_str)))
    eval_result.update(json.loads(get_context_recall(row_str)))
    eval_result = json.dumps(eval_result)
    return eval_result


@cachew(cache_path=EVAL_CACHE, logger=logger)
def process_row(
    idx: Hashable, row_str: str, application: str = "api-eval-bharat"
) -> str:
    eval_record = get_eval_record(row_str, application=application)
    eval_row = evaluate_row(idx, eval_record)
    return eval_row


def main():
    eval_results = []

    df = pd.read_json(
        "data/eval/wandbot_cleaned_annotated_dataset_11-12-2023.jsonl",
        lines=True,
        orient="records",
    )
    correct_df = df[
        (df["is_wandb_query"] == "YES") & (df["correctness"] == "correct")
    ]

    with open(
        "data/eval/wandbot-gpt-4-1106-preview-eval-v1-1.jsonl", "w+"
    ) as outfile:
        for idx, row in tqdm(correct_df.iterrows(), total=len(correct_df)):
            try:
                row_str = row.to_json()
                eval_row = process_row(
                    idx,
                    row_str,
                    application="wandbot-gpt-4-1106-preview-eval-v1.1-bharat",
                )
                outfile.write(eval_row + "\n")
                eval_results.append(eval_row)
            except Exception as e:
                print(e)
                print(idx)


if __name__ == "__main__":
    main()
