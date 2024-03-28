import re
import json
from typing import Any, Hashable

import asyncio
import httpx
import wandb
import pandas as pd
import aiofiles
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
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


# @cachew(cache_path=EVAL_CACHE, logger=logger)
async def get_answer(question: str, application: str = "api-eval-bharat") -> str:
    url = "http://0.0.0.0:8000/chat/query"
    payload = {
        "question": question,
        "chat_history": [],
        "application": application,
        "language": "en",
    }
    async with httpx.AsyncClient(timeout=150.0) as client:
        response = await client.post(url, data=json.dumps(payload))
        response_json = response.json()
    return json.dumps(response_json)

context_metadata_pattern = r"\n?source: .+?\nsource_type: .+?\nhas_code: .+?\n"


def get_individual_contexts(source_documents: str) -> list[str]:
    source_documents = source_documents.split("---")
    source_documents = [
        re.sub(context_metadata_pattern, "", source) for source in source_documents
    ]
    return source_documents


async def get_eval_record(row_str: str, application: str = "api-eval-bharat") -> str:
    row = json.loads(row_str)
    response = await get_answer(row["question"], application=application)
    response = json.loads(response)
    response["ground_truths"] = row["answer"]
    response["reference_notes"] = row["notes"]
    response["contexts"] = [
        "Source: " + source + "\n"  + context
        for source, context in zip(response["sources"].split("\n"), get_individual_contexts(response["source_documents"]))
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


# @cachew(cache_path=EVAL_CACHE, logger=logger)
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


# @cachew(cache_path=EVAL_CACHE, logger=logger)
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


# @cachew(cache_path=EVAL_CACHE, logger=logger)
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


# @cachew(cache_path=EVAL_CACHE, logger=logger)
async def evaluate_row(idx: Hashable, row_str: str) -> str:
    eval_result = {"idx": idx}
    row = json.loads(row_str)
    eval_result.update(row)

    eval_result.update(json.loads(await get_answer_correctness(row_str)))
    eval_result.update(json.loads(await get_answer_relevancy(row_str)))
    eval_result.update(json.loads(await get_answer_faithfulness(row_str)))

    eval_result = json.dumps(eval_result)
    return eval_result


async def main():
    eval_artifact = wandb.Api().artifact("wandbot/wandbot-eval/autoeval_dataset:v3")
    eval_artifact_dir = eval_artifact.download(root="data/eval")

    df = pd.read_json(
        "data/eval/wandbot_cleaned_annotated_dataset_11-12-2023.jsonl",
        lines=True,
        orient="records",
    )
    correct_df = df[
        (df["is_wandb_query"] == "YES") & (df["correctness"] == "correct")
    ]

    async with aiofiles.open(
        "data/eval/baselinev1_1_async.jsonl", "w+"
    ) as outfile:
        for idx, row in tqdm(correct_df.iterrows(), total=len(correct_df)):
            row_str = row.to_json()
            response = await get_eval_record(row_str, application="test-baseline-ayush")
            logger.info("Generated response for idx: %s", idx)
            eval_row = await evaluate_row(idx, response)
            logger.info("Evaluated response for idx: %s", idx)
            try:
                json.loads(eval_row)
                await outfile.write(eval_row + "\n")
            except json.JSONDecodeError:
                logger.error("Failed to parse response for idx: %s", idx)
                continue

if __name__ == "__main__":
    asyncio.run(main())
