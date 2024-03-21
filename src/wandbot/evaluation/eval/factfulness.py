import asyncio
import re
from typing import Any, Optional, Sequence

from llama_index.legacy.evaluation import CorrectnessEvaluator, EvaluationResult

from wandbot.evaluation.eval.utils import (
    make_eval_template,
    safe_parse_eval_response,
)

SYSTEM_TEMPLATE = """You are a Weight & Biases support expert tasked with evaluating the factful consistency of answers to questions asked by users to a technical support chatbot.

You are given the following information:
- a user query,
- the documentation used to generate the answer
- a generated answer.


Your job is to judge the factful consistency of the generated answer with respect to the document.
- An answer is considered factually consistent if it contents can be inferred solely from the provided documentation.
- if an answer contains true information, if the information is not found in the document, then the answer is factually inconsistent.
- The generated answer must provide only correct information according to the documentation.
- Output a score and a decision that represents a holistic evaluation of the generated answer.
- You must return your response only in the below mentioned format. Do not return answers in any other format.

Follow these guidelines for scoring:
- Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.
- If the generated answer is not factually consistent with the document, you should give a score of 1.
- If the generated answer is factually consistent with the document but contains mistakes, you should give a score of 2.
- If the generated answer is factually consistent with the document contents, you should give a score of 3.

Output your final verdict by strictly following JSON format:
{{
    "reason": <<Provide a brief explanation for your decision here>>,
    "score": <<Provide a score as per the above guidelines>>,
    "decision": <<Provide your final decision here, either 'consistent', or 'inconsistent'>>

}}

Example Response 1:
{{
    "reason": "The generated answer the is directly supported by the information present in document and is completely grounded in the document's contents",
    "score": 3,
    "decision": "consistent"
}}

Example Response 2:
{{
    "reason": "The generated answer deviates significantly from the information present in the document and is not based on the document's contents",
    "score": 1,
    "decision": "inconsistent"
}}

Example Response 3:
{{
    "reason": "The generated answer shows methods not present in the provided documentation. The documentation does not mention these methods, thus the answer contains information that cannot be inferred from the provided documentation.",
    "score": 2,
    "decision": "inconsistent"
}}
"""

USER_TEMPLATE = """
## User Query
{query}

## Documentation
{context_str}

## Generated Answer
{generated_answer}
"""

FACTFULNESS_EVAL_TEMPLATE = make_eval_template(SYSTEM_TEMPLATE, USER_TEMPLATE)


class WandbFactfulnessEvaluator(CorrectnessEvaluator):
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
            context_str=re.sub(
                "\n+", "\n", "\n---\n".join(contexts) if contexts else ""
            ),
        )

        passing, reasoning, score = await safe_parse_eval_response(
            eval_response, "consistent"
        )

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=score,
            feedback=reasoning,
        )
