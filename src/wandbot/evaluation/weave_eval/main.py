import os
os.environ["WANDB_ENTITY"] = "wandbot"

import time
import json
import httpx
import wandb
import weave
import asyncio
import pandas as pd
from weave import Evaluation
from llama_index.llms.openai import OpenAI

from wandbot.evaluation.config import EvalConfig
from wandbot.utils import get_logger

from wandbot.evaluation.eval.correctness import (
    CORRECTNESS_EVAL_TEMPLATE,
    WandbCorrectnessEvaluator,
)

logger = get_logger(__name__)
config = EvalConfig()

correctness_evaluator = WandbCorrectnessEvaluator(
    llm=OpenAI(config.eval_judge_model),
    eval_template=CORRECTNESS_EVAL_TEMPLATE,
)

wandb_project = config.wandb_project
wandb_entity = config.wandb_entity

weave.init("weave_test_eval")


@weave.op()
async def get_answer(question: str, application: str = "api-eval") -> str:
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


@weave.op()
async def get_eval_record(row_str: str) -> str:
    row = json.loads(row_str)
    response = await get_answer(row["question"])
    response = json.loads(response)
    response["ground_truths"] = row["answer"]
    response["reference_notes"] = row["notes"]
    response["contexts"] = response["source_documents"]
    response = json.dumps(response)
    return response


@weave.op()
def parse_answer_eval(metric: str, row):
    print("result is: ", row.get("passing"), type(row.get("passing")))
    return {
        f"{metric}_result": True if row.get("passing") is True else False,
    }


@weave.op()
async def get_answer_correctness(model_output: str) -> str:
    row = json.loads(model_output)
    result = await correctness_evaluator.aevaluate(
        query=row["question"],
        response=row["answer"],
        reference=row["ground_truths"],
        contexts=row["contexts"],
        reference_notes=row["reference_notes"],
    )
    # result = parse_answer_eval("answer_correctness", result.dict())
    # result = json.dumps(result)
    return {"answer_correctness": result.dict()["passing"]}
    # return result


dataset_ref = weave.ref(
    "weave:///wandbot/wandbot-eval/object/wandbot_eval_data:eCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU"
).get()
question_rows = dataset_ref.rows
question_rows = [
    {
        "row_str": json.dumps(row)
    } for row in question_rows
]
logger.info("Number of evaluation samples: %s", len(question_rows))

evaluation = Evaluation(
    dataset=question_rows, scorers=[get_answer_correctness]
)

if __name__ == "__main__":
    asyncio.run(evaluation.evaluate(get_eval_record))
