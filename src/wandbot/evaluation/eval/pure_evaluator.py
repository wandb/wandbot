import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple
import regex as re
from openai import AsyncOpenAI

@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    query: str
    response: str
    passing: Optional[bool] = None
    score: Optional[float] = None
    feedback: Optional[str] = None

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
{
    "reason": <<Provide a brief explanation for your decision here>>,
    "score": <<Provide a score as per the above guidelines>>,
    "decision": <<Provide your final decision here, either 'correct', or 'incorrect'>>
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

class PureCorrectnessEvaluator:
    """Evaluates the correctness of a question answering system.
    
    This evaluator depends on a reference answer being provided, in addition to the
    query string and response string. It outputs a score between 1 and 3, where 1 
    is the worst and 3 is the best, along with a reasoning for the score.
    """
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.1,
        score_threshold: float = 2.0,
    ):
        """Initialize the evaluator.
        
        Args:
            openai_client: AsyncOpenAI client instance
            model: OpenAI model to use
            temperature: Temperature for model sampling
            score_threshold: Score threshold for passing evaluation
        """
        self.client = openai_client
        self.model = model
        self.temperature = temperature
        self.score_threshold = score_threshold
        
    async def _get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Get completion from OpenAI API."""
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    async def safe_parse_eval_response(
        self, eval_response: str, expected_decision: str
    ) -> Tuple[bool, str, float]:
        """Safely parse the evaluation response."""
        try:
            result = json.loads(eval_response)
            passing = result["decision"].lower() == expected_decision.lower()
            reasoning = result["reason"]
            score = float(result["score"])
            return passing, reasoning, score
        except (json.JSONDecodeError, KeyError) as e:
            return False, f"Failed to parse evaluation response: {str(e)}", 1.0

    async def evaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        reference: Optional[str] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the correctness of a response.
        
        Args:
            query: The user's question
            response: The generated answer to evaluate
            contexts: List of context documents used
            reference: The reference answer to compare against
            sleep_time_in_seconds: Time to sleep before evaluation
            **kwargs: Additional arguments (reference_notes etc)
            
        Returns:
            EvaluationResult containing the evaluation details
        """
        await asyncio.sleep(sleep_time_in_seconds)

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

        eval_response = await self._get_completion(SYSTEM_TEMPLATE, user_prompt)
        passing, reasoning, score = await self.safe_parse_eval_response(eval_response, "correct")

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=score,
            feedback=reasoning,
        ) 