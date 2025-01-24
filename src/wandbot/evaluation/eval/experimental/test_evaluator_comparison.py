import os
import pytest
import asyncio
import nest_asyncio
import time
import logging
import sys
import json
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from typing import List, Tuple
from llama_index.core.evaluation import EvaluationResult

from wandbot.evaluation.eval.experimental.pure_evaluator import PureCorrectnessEvaluator
from wandbot.evaluation.eval.experimental.llamaindex_evaluator import LlamaIndexCorrectnessEvaluator
from wandbot.evaluation.eval.experimental.test_pure_evaluator import TEST_CASES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,  # Ensure output goes to stdout for pytest capture
    force=True  # Override any existing logging configuration
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dot_env_path = os.path.join(os.path.dirname(__file__), '../../../../../.env')
load_dotenv(dotenv_path=dot_env_path, override=True)

# Allow nested event loops
nest_asyncio.apply()

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

CRITICAL: Return ONLY a JSON object. Do not include ANY text before or after the JSON. No explanations, no notes, just the JSON object:
{
    "reason": "Your explanation here",
    "score": A number between 1 and 3,
    "decision": Either "correct" or "incorrect"
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

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for the test session."""
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"test_logs_{timestamp}.txt"
    
    # Ensure all handlers are removed to prevent duplicate logging
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    
    root.setLevel(logging.INFO)
    logging.info(f"Logging to file: {log_file}")

@pytest.fixture(scope="session")
def openai_client():
    load_dotenv()
    return AsyncOpenAI()

@pytest.fixture(scope="session", autouse=True)
def setup_evaluators(openai_client):
    pure_evaluator = PureCorrectnessEvaluator(
        openai_client=openai_client,
        model="gpt-4-1106-preview",
        temperature=0.1,
        system_template=SYSTEM_TEMPLATE,
    )
    llamaindex_evaluator = LlamaIndexCorrectnessEvaluator(
        llm=LlamaOpenAI(model="gpt-4-1106-preview", temperature=0.1),
        system_template=SYSTEM_TEMPLATE,
    )
    return pure_evaluator, llamaindex_evaluator

async def evaluate_test_case(
    pure_evaluator: PureCorrectnessEvaluator,
    llamaindex_evaluator: LlamaIndexCorrectnessEvaluator,
    query: str,
    response: str,
    reference: str,
    contexts: List[str] = [],
    reference_notes: str = "",
) -> Tuple[EvaluationResult, EvaluationResult]:
    """Evaluate a test case using both evaluators."""
    pure_result = await pure_evaluator.evaluate(
        query=query,
        response=response,
        reference=reference,
        contexts=contexts,
        reference_notes=reference_notes,
    )
    
    # Try up to 3 times for LlamaIndex evaluator
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            llama_result = await llamaindex_evaluator.aevaluate(
                query=query,
                response=response,
                reference=reference,
                contexts=contexts,
                reference_notes=reference_notes,
            )
            # If we get here, the evaluation was successful
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: LlamaIndex evaluator failed to parse response: {str(e)}")
            if attempt == max_retries - 1:
                # On last attempt, create a failed result
                logger.error(f"All {max_retries} attempts failed for LlamaIndex evaluator. Last error: {str(last_error)}")
                llama_result = EvaluationResult(
                    passing=False,
                    score=1.0,
                    feedback=f"Failed to parse evaluation response: {str(last_error)}"
                )
            else:
                # Add a small delay before retrying, increasing with each attempt
                await asyncio.sleep(1 * (attempt + 1))
                continue
        except Exception as e:
            # Log any other unexpected errors and continue
            logger.error(f"Unexpected error from LlamaIndex evaluator: {str(e)}")
            llama_result = EvaluationResult(
                passing=False,
                score=1.0,
                feedback=f"Unexpected error during evaluation: {str(e)}"
            )
            break
    
    return pure_result, llama_result

@pytest.mark.asyncio
async def test_evaluator_comparison(setup_evaluators):
    """Compare results between pure OpenAI and LlamaIndex evaluators."""
    pure_evaluator, llamaindex_evaluator = setup_evaluators
    
    total_cases = len(TEST_CASES)
    logger.info(f"Starting comparison of {total_cases} test cases with max 5 concurrent tasks...")
    
    # Process test cases in batches of 5
    results = []
    for i in range(0, total_cases, 5):
        batch = TEST_CASES[i:i+5]
        logger.info(f"Processing batch of {len(batch)} cases (cases {i+1}-{min(i+5, total_cases)})...")
        
        # Create tasks for current batch
        tasks = []
        for idx, test_case in enumerate(batch, i+1):
            logger.info(f"Starting case {idx}: {test_case['query']}")
            logger.info(f"Case {idx}: Getting pure evaluator result...")
            task = evaluate_test_case(pure_evaluator, llamaindex_evaluator, test_case["query"], test_case["response"], test_case["reference"], test_case["contexts"], test_case["reference_notes"])
            tasks.append(task)
        
        # Run current batch of tasks
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process batch results
        for idx, result in enumerate(batch_results, i+1):
            if isinstance(result, Exception):
                logger.error(f"Error in case {idx}: {str(result)}")
                continue
                
            pure_result, llama_result = result
            
            # Log results for this case
            logger.info(f"\nResults for case {idx}:")
            logger.info(f"Query: {TEST_CASES[idx-1]['query']}")
            logger.info("Pure Evaluator:")
            logger.info(f"  Passing: {pure_result.passing}")
            logger.info(f"  Score: {pure_result.score}")
            logger.info(f"  Reasoning: {pure_result.feedback}")
            logger.info("LlamaIndex Evaluator:")
            logger.info(f"  Passing: {llama_result.passing}")
            logger.info(f"  Score: {llama_result.score}")
            logger.info(f"  Reasoning: {llama_result.feedback}")
            logger.info("Agreement:")
            logger.info(f"  Same passing result: {pure_result.passing == llama_result.passing}")
            logger.info(f"  Score difference: {abs(pure_result.score - llama_result.score)}")
            
            # Store results
            results.append({
                "query": TEST_CASES[idx-1]["query"],
                "pure": {
                    "passing": pure_result.passing,
                    "score": pure_result.score,
                    "feedback": pure_result.feedback
                },
                "llama": {
                    "passing": llama_result.passing,
                    "score": llama_result.score,
                    "feedback": llama_result.feedback
                },
                "agreement": {
                    "passing": pure_result.passing == llama_result.passing,
                    "score_diff": abs(pure_result.score - llama_result.score)
                }
            })
        
        # Add a small delay between batches to avoid rate limits
        if i + 5 < total_cases:
            await asyncio.sleep(1)
    
    # Analyze results
    passing_agreement = sum(1 for r in results if r["agreement"]["passing"])
    score_diffs = [r["agreement"]["score_diff"] for r in results]
    avg_score_diff = sum(score_diffs) / len(score_diffs)
    
    logger.info(f"\nFinal Agreement Analysis:")
    logger.info(f"Total test cases processed: {total_cases}")
    logger.info(f"Cases with same passing result: {passing_agreement} ({passing_agreement/total_cases*100:.1f}%)")
    logger.info(f"Average score difference: {avg_score_diff:.2f}")
    
    # List cases where evaluators disagreed
    disagreements = [r for r in results if not r["agreement"]["passing"]]
    if disagreements:
        logger.warning("\nCases where evaluators disagreed:")
        for case in disagreements:
            logger.warning(f"\nQuery: {case['query']}")
            logger.warning(f"Pure evaluator: passing={case['pure']['passing']}, score={case['pure']['score']}")
            logger.warning(f"LlamaIndex: passing={case['llama']['passing']}, score={case['llama']['score']}")
    
    # Assert reasonable agreement between evaluators
    min_agreement_pct = 1.0  # At least 95% agreement on passing/failing
    max_avg_score_diff = 1.0  # Average score difference should be less than 1.0
    
    agreement_pct = passing_agreement / total_cases
    assert agreement_pct >= min_agreement_pct, \
        f"Evaluators only agree on {agreement_pct*100:.1f}% of cases (minimum {min_agreement_pct*100}% required)"
    
    assert avg_score_diff <= max_avg_score_diff, \
        f"Average score difference {avg_score_diff:.2f} exceeds maximum allowed {max_avg_score_diff}"

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__])) 