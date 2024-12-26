import json
import httpx
import weave
import asyncio
import re
from weave import Evaluation
from weave import Model
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

weave.init(f"{wandb_entity}/{wandb_project}")


@weave.op
async def get_answer(question: str, application: str = "api-eval") -> str:
    url = "http://0.0.0.0:8000/chat/query"
    payload = {
        "question": question,
        "application": application,
        "language": config.language,
    }
    async with httpx.AsyncClient(timeout=900.0) as client:
        response = await client.post(url, json=payload)
        response_json = response.json()
    return json.dumps(response_json)


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
async def get_eval_record(
    question: str,
) -> dict:
    response = await get_answer(question)
    response = json.loads(response)
    return {
        "system_prompt": response["system_prompt"],
        "generated_answer": response["answer"],
        "retrieved_contexts_individual": parse_text_to_json(
            response["source_documents"]
        ),
        "model": response["model"],
        "total_tokens": response["total_tokens"],
        "prompt_tokens": response["prompt_tokens"],
        "completion_tokens": response["completion_tokens"],
        "time_taken": response["time_taken"],
    }


class EvaluatorModel(Model):
    eval_judge_model: str = config.eval_judge_model

    @weave.op
    async def predict(self, question: str) -> dict:
        # Model logic goes here
        prediction = await get_eval_record(question)
        return prediction


@weave.op
async def get_answer_correctness(
    question: str, ground_truth: str, notes: str, model_output: dict
) -> dict:
    result = await correctness_evaluator.aevaluate(
        query=question,
        response=model_output["generated_answer"],
        reference=ground_truth,
        contexts=model_output["retrieved_contexts"],
        reference_notes=notes,
    )
    return {"answer_correctness": result.dict()["passing"]}


dataset_ref = weave.ref(config.eval_dataset).get()
question_rows = dataset_ref.rows
question_rows = [
    {
        "question": row["question"],
        "ground_truth": row["answer"],
        "notes": row["notes"],
    }
    for row in question_rows
]
logger.info("Number of evaluation samples: %s", len(question_rows))

evaluation = Evaluation(dataset=question_rows, scorers=[get_answer_correctness])
if __name__ == "__main__":
    with weave.attributes(
        {"evaluation_strategy_name": config.evaluation_strategy_name}
    ):
        asyncio.run(evaluation.evaluate(EvaluatorModel()))
