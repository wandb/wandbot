import asyncio
from typing import Any, Optional, Sequence

from llama_index.legacy.evaluation import CorrectnessEvaluator, EvaluationResult

from wandbot.evaluation.eval.utils import (
    make_eval_template,
    safe_parse_eval_response,
)

SYSTEM_TEMPLATE = """You are a Weight & Biases support expert tasked with evaluating the relevancy of answers to questions asked by users to a technical support chatbot.

You are given the following information:
- a user query,
- a reference answer
- a generated answer.


Your job is to judge the relevance the generated answer to the user query.
- Consider whether the answer addresses all aspects of the question and aligns with the user's intent and provides appropriate and on-topic response.
- Measure the generated answer on its sensibleness, meaning it needs to make sense in context and be specific i.e. it is comprehensive without being too vague.
- Compare the generated answer to the reference answer for its relevancy, sensibleness and specificity.
- Output a score and a decision that represents a holistic evaluation of the generated answer.
- You must return your response only in the below mentioned format. Do not return answers in any other format.

Follow these guidelines for scoring:
- Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes or lacks specificity, you should give a score of 2.
- If the generated answer is relevant and comprehensive, you should give a score of 3.

Output your final verdict by strictly following JSON format:
{{
    "reason": <<Provide a brief explanation for your decision here>>,
    "score": <<Provide a score as per the above guidelines>>,
    "decision": <<Provide your final decision here, either 'relevant', or 'irrelevant'>>

}}

Example Response 1:
{{
    "reason": "The generated answer is relevant and provides a similar level of detail as the reference answer. It also provides information that is relevant to the user's query.",
    "score": 3,
    "decision": "relevant"
}}

Example Response 2:
{{
    "reason": "The generated answer deviates significantly from the reference answer, and is not directly answering the user's query",
    "score": 1,
    "decision": "irrelevant"
}}

Example Response 3:
{{
    "reason": "The generated answer is relevant and provides a similar level of detail as the reference answer. However, it introduces variations in the code example that are not mentioned in the documentation. This could potentially confuse users if the method is not part of the documented API.
    "score": 2,
    "decision": "irrelevant"
}}
"""

USER_TEMPLATE = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

RELEVANCY_EVAL_TEMPLATE = make_eval_template(SYSTEM_TEMPLATE, USER_TEMPLATE)


class WandbRelevancyEvaluator(CorrectnessEvaluator):
    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None or reference is None:
            print(query, response, reference, flush=True)
            raise ValueError("query, response, and reference must be provided")

        eval_response = await self._service_context.llm.apredict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference,
        )

        passing, reasoning, score = await safe_parse_eval_response(
            eval_response, "relevant"
        )

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=score,
            feedback=reasoning,
        )
