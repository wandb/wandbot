import asyncio
from typing import Any, Optional, List, Tuple, Dict
import json
import regex as re
from llama_index.core.evaluation import CorrectnessEvaluator, EvaluationResult
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

from wandbot.evaluation.eval.utils import (
    make_eval_template,
    safe_parse_eval_response,
)

SYSTEM_TEMPLATE = """You are a Weight & Biases support expert tasked with evaluating the correctness of answers to questions asked by users to a technical support chatbot.

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

CRITICAL: You must output ONLY a JSON object. No text before or after. No explanations. No notes. Just the JSON object in exactly this format:
{
    "reason": <<Provide a brief explanation for your decision here>>,
    "score": <<Provide a score as per the above guidelines>>,
    "decision": <<Provide your final decision here, either 'correct', or 'incorrect'>>
}

Follow these guidelines for scoring:
- Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.
- If the generated answer is not correct in comparison to the reference, you should give a score of 1.
- If the generated answer is correct in comparison to the reference but contains mistakes, you should give a score of 2.
- If the generated answer is correct in comparison to the reference and completely answer's the user's query, you should give a score of 3.

Example Response 1:
{
    "reason": "The generated answer has the exact details as the reference answer and completely answer's the user's query.",
    "score": 3,
    "decision": "correct"
}

Example Response 2:
{
    "reason": "The generated answer doesn't match the reference answer, and deviates from the documentation provided",
    "score": 1,
    "decision": "incorrect"
}

Example Response 3:
{
    "reason": "The generated answer follows the same steps as the reference answer. However, it includes assumptions about methods that are not mentioned in the documentation.",
    "score": 2,
    "decision": "incorrect"
}"""

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

class LlamaIndexCorrectnessEvaluator(CorrectnessEvaluator):
    def __init__(self, llm, system_template: Optional[str] = None):
        super().__init__(llm)
        self.system_template = system_template or SYSTEM_TEMPLATE
        self.eval_template = ChatPromptTemplate([
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_template),
            ChatMessage(role=MessageRole.USER, content=USER_TEMPLATE)
        ])

    def safe_parse_eval_response(self, response: str) -> Dict[str, Any]:
        """Safely parse the evaluation response."""
        try:
            # Remove any backticks and 'json' language specifier
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.split("\n", 1)[1]  # Remove first line with ```json
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response.rsplit("\n", 1)[0]  # Remove last line with ```
            cleaned_response = cleaned_response.strip()
            
            # Parse the JSON response
            parsed_response = json.loads(cleaned_response)
            
            # Validate required fields
            if not all(key in parsed_response for key in ["reason", "score", "decision"]):
                raise ValueError("Missing required fields in response")
            
            return parsed_response
        except Exception as e:
            print(f"Failed to parse response: {str(e)}\nResponse was: {response}", flush=True)
            return {
                "reason": "Failed to parse evaluation response: " + str(e),
                "score": 1,
                "decision": "incorrect"
            }

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[List[str]] = [],
        reference: Optional[str] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None or reference is None:
            print(query, response, reference, flush=True)
            raise ValueError("query, response, and reference must be provided")

        formatted_messages = self.eval_template.format_messages(
            query=query,
            generated_answer=response,
            reference_answer=reference,
            context_str=re.sub(
                "\n+", "\n", "\n---\n".join(contexts) if contexts else ""
            ),
            reference_notes=kwargs.get("reference_notes", ""),
        )

        eval_response = await self._llm.achat(
            messages=formatted_messages
        )

        print(f"Raw LLM response: {eval_response.message.content}", flush=True)

        parsed_response = self.safe_parse_eval_response(eval_response.message.content)

        return EvaluationResult(
            query=query,
            response=response,
            passing=parsed_response["decision"].lower() == "correct",
            score=float(parsed_response["score"]),
            feedback=parsed_response["reason"],
        ) 