from typing import Any, Hashable, Tuple

import dotenv

dotenv.load_dotenv()

import json

import nest_asyncio
import pandas as pd
import requests
from cachew import cachew
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

nest_asyncio.apply()


EVAL_CACHE = "data/cache/eval_cache"
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


@cachew(cache_path=EVAL_CACHE)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_answer(
    question: str, application: str = "api-eval-bharat"
) -> dict[str, Any]:
    url = "http://0.0.0.0:8000/query"
    payload = {
        "question": question,
        "language": "en",
        "application": application,
    }
    response = requests.post(url, data=json.dumps(payload))
    return response.json()


@cachew(cache_path=EVAL_CACHE)
def get_eval_record(
    row: pd.Series, application: str = "api-eval-bharat"
) -> dict[str, Any]:
    row = row.to_dict()
    response = get_answer(row["question"], application=application)
    response["ground_truths"] = row["answer"]
    response["reference_notes"] = row["notes"]
    response["contexts"] = [
        "\nSource: " + source["source"] + " \n " + source["text"]
        for source in json.loads(response["source_documents"])
    ]
    return response


def parse_answer_eval(metric: str, row: dict[str, Any]) -> dict[str, Any]:
    result = {
        f"{metric}_score": row.get("score"),
        f"{metric}_result": row.get("passing"),
        f"{metric}_reason": row.get("feedback"),
    }
    return result


@cachew(cache_path=EVAL_CACHE)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_answer_correctness(row: dict[str, Any]) -> dict[str, Any]:
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
    return result


@cachew(cache_path=EVAL_CACHE)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_answer_relevancy(row: dict[str, Any]) -> dict[str, Any]:
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
    return result


@cachew(cache_path=EVAL_CACHE)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_answer_faithfulness(row: dict[str, Any]) -> dict[str, Any]:
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
    return result


@cachew(cache_path=EVAL_CACHE)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_answer_similarity(row: dict[str, Any]) -> dict[str, Any]:
    result = metrics.answer_similarity.score_single(row)
    return {"answer_similarity_score_(ragas)": result}


@cachew(cache_path=EVAL_CACHE)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_context_precision(row: dict[str, Any]) -> dict[str, Any]:
    result = metrics.context_precision.score_single(row)
    return {"context_precision_score": result}


@cachew(cache_path=EVAL_CACHE)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_context_recall(row: dict[str, Any]) -> dict[str, Any]:
    result = metrics.context_recall.score_single(row)
    return {"context_recall_score": result}


@cachew(cache_path=EVAL_CACHE)
def evaluate_row(idx: Hashable, row: dict[str, Any]) -> dict[str, Any]:
    eval_result = {"idx": idx}
    eval_result.update(row)
    eval_result.update(get_answer_correctness(row))
    eval_result.update(get_answer_relevancy(row))
    eval_result.update(get_answer_faithfulness(row))
    eval_result.update(get_answer_similarity(row))
    eval_result.update(get_context_precision(row))
    eval_result.update(get_context_recall(row))
    return eval_result


@cachew(cache_path=EVAL_CACHE)
def process_row(
    args: Tuple[Hashable, pd.Series], application: str = "api-eval-bharat"
) -> str:
    idx, row = args
    eval_record = get_eval_record(row, application=application)
    eval_row = evaluate_row(idx, eval_record)
    result = json.dumps(eval_row)
    return result


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

    with open("data/wandbot-gpt-4-eval.jsonl", "w+") as outfile:
        for idx, row in tqdm(
            correct_df.sample(frac=1).iterrows(), total=len(correct_df)
        ):
            try:
                eval_row = process_row(
                    (idx, row), application="gpt-4-eval-bharat"
                )
                outfile.write(eval_row + "\n")
                eval_results.append(eval_row)
            except:
                print(idx)


if __name__ == "__main__":
    main()
