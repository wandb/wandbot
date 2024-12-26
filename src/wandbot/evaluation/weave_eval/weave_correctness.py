import asyncio
from typing import Any, Optional, Sequence

import regex as re
from llama_index.core.evaluation import CorrectnessEvaluator, EvaluationResult

from wandbot.evaluation.eval.utils import (
    make_eval_template,
    safe_parse_eval_response,
)

import weave

SYSTEM_TEMPLATE = """You are a Weight & Biases support expert tasked with evaluating the correctness of answers to questions asked by users to a a technical support chatbot.

You are given the following information:
- a user query,
- the documentation used to generate the answer
- a reference answer
- the reason why the reference answer is correct, and
- a generated answer.


Your job is to judge the relevance and correctness of the generated answer.
- Consider whether the answer addresses all aspects of the question.
- The generated answer must provide only correct information according to the documentation.
- Compare the generated answer to the reference answer for completeness and correctness.
- Output a score and a decision that represents a holistic evaluation of the generated answer.
- You must return your response only in the below mentioned format. Do not return answers in any other format.

Follow these guidelines for scoring:
- Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.
- If the generated answer is not correct in comparison to the reference, you should give a score of 1.
- If the generated answer is correct in comparison to the reference but contains mistakes, you should give a score of 2.
- If the generated answer is correct in comparision to the reference and completely answer's the user's query, you should give a score of 3.

Output your final verdict by strictly following JSON format:
{{
    "reason": <<Provide a brief explanation for your decision here>>,
    "score": <<Provide a score as per the above guidelines>>,
    "decision": <<Provide your final decision here, either 'correct', or 'incorrect'>>

}}

Example Response 1:
{{
    "reason": "The generated answer has the exact details as the reference answer and completely answer's the user's query.",
    "score": 3,
    "decision": "correct"
}}

Example Response 2:
{{
    "reason": "The generated answer doesn't match the reference answer, and deviates from the documentation provided",
    "score": 1,
    "decision": "incorrect"
}}

Example Response 3:
{{
    "reason": "The generated answer follows the same steps as the reference answer. However, it includes assumptions about methods that are not mentioned in the documentation.",
    "score": 2,
    "decision": "incorrect"
}}
"""


USER_TEMPLATE = """
## User Query
{query}

## Documentation
{context_str}

## Reference Answer
{reference_answer}

## Reference Correctness Reason
{reference_notes}

## Generated Answer
{generated_answer}
"""

CORRECTNESS_EVAL_TEMPLATE = make_eval_template(SYSTEM_TEMPLATE, USER_TEMPLATE)


class WandbCorrectnessEvaluator(CorrectnessEvaluator):
    @weave.op()
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

        eval_response = await self._llm.apredict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference,
            context_str=re.sub(
                "\n+", "\n", "\n---\n".join(contexts) if contexts else ""
            ),
            reference_notes=kwargs.get("reference_notes", ""),
        )

        passing, reasoning, score = await safe_parse_eval_response(
            eval_response, "correct"
        )

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=score,
            feedback=reasoning,
        )