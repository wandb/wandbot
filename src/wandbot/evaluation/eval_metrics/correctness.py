import json
from typing import Any, Optional, List, Tuple
import regex as re
from pydantic import BaseModel, Field

from wandbot.evaluation.utils.utils import EvaluationResult
from wandbot.models.llm import LLMModel

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

Follow these guidelines for scoring:
- Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.
- If the generated answer is not correct in comparison to the reference, you should give a score of 1.
- If the generated answer is correct in comparison to the reference but contains mistakes, you should give a score of 2.
- If the generated answer is correct in comparison to the reference and completely answer's the user's query, you should give a score of 3.

CRITICAL: You must output ONLY a JSON object. No text before or after. No explanations. No notes. Just the JSON object in exactly this format:
{
    "reason": <<Provide a brief explanation for your decision here>>,
    "score": <<Provide a score as per the above guidelines>>,
    "decision": <<Provide your final decision here, either 'correct', or 'incorrect'>>
}

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


class CorrectnessEvaluationResult(BaseModel):
    reason: str = Field(..., description="Provide a brief explanation for your decision here")
    score: float = Field(..., description="Provide a score as per the above guidelines")
    decision: str = Field(..., description="Provide your final decision here, either 'correct', or 'incorrect'")


class WandBotCorrectnessEvaluator:
    """Evaluates the correctness of a question answering system.
    
    This evaluator depends on a reference answer being provided, in addition to the
    query string and response string. It outputs a score between 1 and 3, where 1 
    is the worst and 3 is the best, along with a reasoning for the score.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4-1106-preview",
        provider: str = "openai",
        temperature: float = 0.1,
        system_template: Optional[str] = None,
        max_concurrent_requests: int = 20,
        **kwargs
    ):
        """Initialize the evaluator.
        
        Args:
            model_name: Name of the model to use
            provider: Provider of the model (e.g., "openai" or "anthropic")
            temperature: Temperature for model sampling
            system_template: Optional custom system template to use for evaluation
            max_concurrent_requests: Maximum number of concurrent requests
            **kwargs: Additional keyword arguments for LLMModel
        """
        self.llm = LLMModel(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            response_model=CorrectnessEvaluationResult,
            n_parallel_api_calls=max_concurrent_requests,
            **kwargs
        )
        self.system_template = system_template or SYSTEM_TEMPLATE

    async def _get_completion(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Get completion from the model."""
        return await self.llm.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )

    async def safe_parse_eval_response(
        self, eval_response: str, expected_decision: str
    ) -> Tuple[bool, str, float, bool, Optional[str]]:
        """Safely parse the evaluation response.
        
        Returns:
            Tuple of (passing, reasoning, score, has_error, error_message)
        """
        try:
            # Clean up the response if it's wrapped in ```json blocks
            cleaned_response = eval_response
            if eval_response.startswith("```json"):
                cleaned_response = eval_response.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_response)
            passing = result["decision"].lower() == expected_decision.lower()
            reasoning = result["reason"]
            score = float(result["score"])
            return passing, reasoning, score, False, None
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Failed to parse evaluation response: {str(e)}"
            return False, "Evaluation failed due to parsing error", 1.0, True, error_msg

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the correctness of a response.
        
        Args:
            query: The user's question
            response: The generated answer to evaluate
            contexts: List of context documents used
            reference: The reference answer to compare against
            **kwargs: Additional arguments (reference_notes etc)
            
        Returns:
            EvaluationResult containing the evaluation details
        """

        try:
            if query is None or response is None or reference is None:
                raise ValueError("query, response, and reference must be provided")

            user_prompt = USER_TEMPLATE.format(
                query=query,
                generated_answer=response,
                reference_answer=reference,
                context_str=re.sub(
                    "\n+", "\n", "\n---\n".join(contexts) if contexts else ""
                ),
                reference_notes=kwargs.get("reference_notes", ""),
            )

            eval_response = await self._get_completion(system_prompt=self.system_template, user_prompt=user_prompt)
            passing, reasoning, score, has_error, error_msg = await self.safe_parse_eval_response(eval_response, "correct")

            if has_error:
                return EvaluationResult(
                    query=query,
                    response=response,
                    passing=passing,
                    score=score,
                    reasoning=reasoning,
                    has_error=True,
                    error_message=error_msg
                )

            return EvaluationResult(
                query=query,
                response=response,
                passing=passing,
                score=score,
                reasoning=reasoning,
                has_error=False,
                error_message=None
            )
        except Exception as e:
            error_msg = f"Error during evaluation: {str(e)}"
            return EvaluationResult(
                query=query or "",
                response=response or "",
                passing=False,
                score=1.0,  # Lowest score since evaluation failed
                reasoning="Evaluation failed due to an error",
                has_error=True,
                error_message=error_msg
            )
