import asyncio
import json
import re
import time
from typing import Any, Hashable

import aiofiles
import httpx
import pandas as pd
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

import wandb
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
from wandbot.utils import get_logger

logger = get_logger(__name__)


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
async def get_answer(
    question: str, application: str = "api-eval-bharat"
) -> str:
    url = "http://0.0.0.0:8000/chat/query"
    payload = {
        "question": question,
        "application": application,
        "language": "en",
    }
    async with httpx.AsyncClient(timeout=200.0) as client:
        response = await client.post(url, json=payload)
        response_json = response.json()
    return json.dumps(response_json)


context_metadata_pattern = r"\n?source: .+?\nsource_type: .+?\nhas_code: .+?\n"


def get_individual_contexts(source_documents: str) -> list[str]:
    source_documents = source_documents.split("---")
    source_documents = [
        re.sub(context_metadata_pattern, "", source)
        for source in source_documents
    ]
    return source_documents


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def get_eval_record(
    row_str: str, application: str = "api-eval-bharat"
) -> str:
    row = json.loads(row_str)
    response = await get_answer(row["question"], application=application)
    response = json.loads(response)
    response["ground_truths"] = row["answer"]
    response["reference_notes"] = row["notes"]
    # response["contexts"] = [
    #     "\nSource: " + source["source"] + " \n " + source["text"]
    #     for source in json.loads(response["source_documents"])
    # ]
    response["contexts"] = response["source_documents"]
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
async def get_answer_correctness(row_str: str) -> str:
    row = json.loads(row_str)
    result = await correctness_evaluator.aevaluate(
        query=row["question"],
        response=row["answer"],
        reference=row["ground_truths"],
        contexts=row["contexts"],
        reference_notes=row["reference_notes"],
    )
    result = parse_answer_eval("answer_correctness", result.dict())
    result = json.dumps(result)
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def get_answer_relevancy(row_str: str) -> str:
    row = json.loads(row_str)
    result = await relevancy_evaluator.aevaluate(
        query=row["question"],
        response=row["answer"],
        contexts=row["contexts"],
        reference=row["ground_truths"],
    )
    result = parse_answer_eval("answer_relevancy", result.dict())
    result = json.dumps(result)
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def get_answer_faithfulness(row_str: str) -> str:
    row = json.loads(row_str)
    result = await faithfulness_evaluator.aevaluate(
        query=row["question"],
        response=row["answer"],
        contexts=row["contexts"],
        reference=row["ground_truths"],
    )

    result = parse_answer_eval("answer_faithfulness", result.dict())
    result = json.dumps(result)
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def evaluate_row(idx: Hashable, row_str: str) -> str:
    eval_result = {"idx": idx}
    row = json.loads(row_str)
    eval_result.update(row)

    eval_result.update(json.loads(await get_answer_correctness(row_str)))
    eval_result.update(json.loads(await get_answer_relevancy(row_str)))
    eval_result.update(json.loads(await get_answer_faithfulness(row_str)))
    await asyncio.sleep(5)
    eval_result = json.dumps(eval_result)
    return eval_result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def process_row(idx, row, outfile):
    """
    Process a chunk of the dataframe asynchronously and write results to the file.
    """
    row_str = row.to_json()
    response = await get_eval_record(row_str, application="test-baseline-ayush")
    logger.info(f"Generated response for idx: {idx}")
    eval_row = await evaluate_row(idx, response)
    logger.info(f"Evaluated response for idx: {idx}")
    try:
        json.loads(eval_row)
        await outfile.write(eval_row + "\n")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse response for idx: {idx}")


def log_eval_result(eval_result_path: str, duration: float) -> None:
    project = "wandbot-eval"
    entity = "wandbot"

    run = wandb.init(project=project, entity=entity)

    eval_df = pd.read_json(eval_result_path, lines=True)
    eval_df = eval_df.sort_values(by="idx").reset_index(drop=True)

    logger.info(f"Number of eval samples: {len(eval_df)}")
    run.log({"Evaluation Results": eval_df})

    score_columns = [col for col in eval_df.columns if col.endswith("_score")]
    mean_scores = eval_df[score_columns].mean()
    mode_scores = eval_df[score_columns].mode()
    percent_grade3 = (eval_df[score_columns] == 3).mean()
    percent_grade2 = (eval_df[score_columns] == 2).mean()
    percent_grade1 = (eval_df[score_columns] == 1).mean()

    # Select columns ending with "_result" and calculate the percentage of True values
    result_columns = [col for col in eval_df.columns if col.endswith("_result")]
    percentage_true_results = (
        eval_df[result_columns].sum() / eval_df[result_columns].count()
    )

    final_eval_results = {}
    final_eval_results.update(mean_scores.to_dict())
    final_eval_results.update(mode_scores.iloc[0].to_dict())
    final_eval_results.update(percent_grade3.to_dict())
    final_eval_results.update(percent_grade2.to_dict())
    final_eval_results.update(percent_grade1.to_dict())
    final_eval_results.update(percentage_true_results.to_dict())

    logger.info(
        f"Final Eval Results: {json.dumps(final_eval_results, indent=4)}"
    )
    run.log(final_eval_results)

    run.summary["duration(s)"] = duration


async def main():
    eval_artifact = wandb.Api().artifact(
        "wandbot/wandbot-eval/autoeval_dataset:v3"
    )
    eval_artifact_dir = eval_artifact.download(root="data/eval")

    df = pd.read_json(
        "data/eval/wandbot_cleaned_annotated_dataset_11-12-2023.jsonl",
        lines=True,
        orient="records",
    )
    correct_df = df[
        (df["is_wandb_query"] == "YES") & (df["correctness"] == "correct")
    ]
    logger.info("Number of evaluation samples: %s", len(correct_df))

    start_time = time.time()

    async with aiofiles.open("data/eval/eval.jsonl", "w+") as outfile:
        tasks = [
            process_row(idx, row, outfile)
            for idx, row in tqdm(correct_df.iterrows())
        ]
        await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Total runtime: {duration:.2f} seconds")

    log_eval_result("data/eval/eval.jsonl", duration)


if __name__ == "__main__":
    asyncio.run(main())
