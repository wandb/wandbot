import os
import pytest
import asyncio
from openai import AsyncOpenAI
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.evaluation import CorrectnessEvaluator

from wandbot.evaluation.eval.pure_evaluator import PureCorrectnessEvaluator
from wandbot.evaluation.eval.correctness import WandBotCorrectnessEvaluator

# Test data
TEST_CASES = [
    {
        "query": "How do I create a new wandb project?",
        "response": "To create a new W&B project, you can initialize it in your code using wandb.init(project='your-project-name'). This will create the project if it doesn't exist.",
        "reference": "You can create a new W&B project by using wandb.init(project='your-project-name') in your code. If the project doesn't exist, it will be created automatically.",
        "reference_notes": "This is the standard way to create projects programmatically",
        "contexts": ["To create a new project, use wandb.init() with the project parameter"],
        "expected_passing": True,
    },
    {
        "query": "How do I delete a run?",
        "response": "You can delete a run by clicking the delete button.",
        "reference": "To delete a run, you can either use the web interface by clicking the delete button on the run page, or programmatically using the wandb.Api().delete_run(path) method.",
        "reference_notes": "Both UI and API methods are valid ways to delete runs",
        "contexts": ["Runs can be deleted via UI or API"],
        "expected_passing": False,
    }
]

@pytest.fixture
def openai_client():
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

@pytest.fixture
def pure_evaluator(openai_client):
    return PureCorrectnessEvaluator(
        openai_client=openai_client,
        model="gpt-4-1106-preview",
        temperature=0.1
    )

@pytest.fixture
def llamaindex_evaluator():
    llm = LlamaOpenAI(model="gpt-4-1106-preview", temperature=0.1)
    return WandBotCorrectnessEvaluator(llm=llm)

@pytest.mark.asyncio
async def test_evaluator_consistency(pure_evaluator, llamaindex_evaluator):
    """Test that both evaluators give consistent results."""
    
    for test_case in TEST_CASES:
        # Evaluate with pure evaluator
        pure_result = await pure_evaluator.evaluate(
            query=test_case["query"],
            response=test_case["response"],
            reference=test_case["reference"],
            contexts=test_case["contexts"],
            reference_notes=test_case["reference_notes"]
        )
        
        # Evaluate with LlamaIndex evaluator
        llama_result = await llamaindex_evaluator.evaluate(
            query=test_case["query"],
            response=test_case["response"],
            reference=test_case["reference"],
            contexts=test_case["contexts"],
            reference_notes=test_case["reference_notes"]
        )
        
        # Compare results
        assert pure_result.passing == llama_result.passing, \
            f"Evaluators disagree on passing for query: {test_case['query']}"
        
        assert abs(pure_result.score - llama_result.score) <= 1.0, \
            f"Scores differ too much for query: {test_case['query']}"
            
        # Check against expected passing
        assert pure_result.passing == test_case["expected_passing"], \
            f"Pure evaluator result doesn't match expected for query: {test_case['query']}"
            
        print(f"\nResults for query: {test_case['query']}")
        print(f"Pure Evaluator: passing={pure_result.passing}, score={pure_result.score}")
        print(f"LlamaIndex: passing={llama_result.passing}, score={llama_result.score}")
        print(f"Pure feedback: {pure_result.feedback}")
        print(f"LlamaIndex feedback: {llama_result.feedback}")

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__])) 