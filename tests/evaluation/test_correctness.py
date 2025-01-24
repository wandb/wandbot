import os
import pytest
from unittest.mock import patch
import sys
from openai import AsyncOpenAI
from dataclasses import dataclass

from wandbot.evaluation.eval.correctness import (
    WandBotCorrectnessEvaluator,
    EvaluationResult
)
from wandbot.evaluation.weave_eval.eval import WandbotCorrectnessScorer
from wandbot.evaluation.eval_config import EvalConfig

# Mock config for basic unit tests
@dataclass
class MockConfig:
    """Mock configuration for testing."""
    wandb_entity: str = "test"
    wandb_project: str = "test"
    eval_judge_model: str = "gpt-4"
    eval_judge_temperature: float = 0.1
    debug: bool = True
    n_debug_samples: int = 5
    n_trials: int = 1
    n_weave_parallelism: int = 1
    wandbot_url: str = "http://localhost:8000"
    lang: str = "en"
    experiment_name: str = "test"
    evaluation_name: str = "test"

@pytest.fixture
def mock_config():
    """Mock configuration for basic unit tests."""
    return MockConfig()

@pytest.fixture
def real_config():
    """Real configuration for integration tests."""
    return EvalConfig(
        lang="en",
        eval_judge_model="gpt-4-1106-preview",
        eval_judge_temperature=0.1,
        experiment_name="test-exp",
        evaluation_name="test-eval",
        n_trials=1,
        n_weave_parallelism=1,
        wandbot_url="http://localhost:8000",
        wandb_entity="test",
        wandb_project="test",
        debug=True,
        n_debug_samples=2
    )

@pytest.fixture(autouse=True)
def setup_test_args():
    """Set up test arguments for config parsing."""
    original_argv = sys.argv
    sys.argv = [
        "pytest",
        "--lang", "en",
        "--eval_judge_model", "gpt-4-1106-preview",
        "--eval_judge_temperature", "0.1",
        "--experiment_name", "test-exp",
        "--evaluation_name", "test-eval",
        "--n_trials", "1",
        "--n_weave_parallelism", "1",
        "--wandbot_url", "http://localhost:8000",
        "--wandb_entity", "test",
        "--wandb_project", "test",
        "--debug", "true",
        "--n_debug_samples", "2"
    ]
    yield
    sys.argv = original_argv

# Test cases with known expected outcomes
TEST_CASES = [
    {
        "query": "How do I log metrics in wandb?",
        "response": "To log metrics in wandb, use wandb.log() with a dictionary of metrics. For example: wandb.log({'loss': 0.2, 'accuracy': 0.85})",
        "reference": "Use wandb.log() to log metrics. Pass a dictionary with metric names as keys and values as the metric values. Example: wandb.log({'loss': 0.2, 'accuracy': 0.85})",
        "reference_notes": "This is the correct way to log metrics in wandb using the wandb.log() method.",
        "contexts": ["wandb.log() is used to log metrics during training. It accepts a dictionary of metric names and values."],
        "expected_score": 3.0,
        "expected_passing": True,
        "reason": "Perfect match with reference"
    },
    {
        "query": "How do I save a model checkpoint?",
        "response": "Just use model.save('checkpoint.h5')",
        "reference": "To save model checkpoints with wandb, use either wandb.save() to save files directly, or wandb.log_artifact() to version and track your models.",
        "reference_notes": "The response should mention wandb-specific methods for saving checkpoints.",
        "contexts": ["Model checkpoints can be saved using wandb.save() or wandb.log_artifact()"],
        "expected_score": 1.0,
        "expected_passing": False,
        "reason": "Incorrect method not using wandb"
    },
    {
        "query": "How do I log images in wandb?",
        "response": "Use wandb.log() with a wandb.Image object",
        "reference": "To log images, use wandb.log() with wandb.Image. You can log PNG, JPG, or other image formats. You can also add optional captions and masks. Example: wandb.log({'image': wandb.Image(img_array, caption='My Image')})",
        "reference_notes": "The response should mention wandb.log with wandb.Image and ideally include format support and optional parameters.",
        "contexts": ["Images can be logged using wandb.log() with wandb.Image objects. Supports various formats and optional parameters like captions."],
        "expected_score": 2.0,
        "expected_passing": False,
        "reason": "Correct but incomplete"
    }
]

# Basic unit tests with mock config
@pytest.mark.usefixtures("mock_config")
class TestWithMockConfig:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_safe_parse_eval_response(self):
        """Test parsing of evaluation responses."""
        evaluator = WandBotCorrectnessEvaluator(client=None)
        
        # Test valid JSON response
        valid_response = '''{"reason": "test", "score": 3, "decision": "correct"}'''
        passing, reasoning, score, has_error, error_msg = await evaluator.safe_parse_eval_response(valid_response, "correct")
        assert passing == True
        assert reasoning == "test"
        assert score == 3.0
        assert has_error == False
        assert error_msg is None
        
        # Test JSON with backticks
        response_with_backticks = '''```json
        {"reason": "test", "score": 2, "decision": "incorrect"}
        ```'''
        passing, reasoning, score, has_error, error_msg = await evaluator.safe_parse_eval_response(response_with_backticks, "correct")
        assert passing == False
        assert reasoning == "test"
        assert score == 2.0
        assert has_error == False
        assert error_msg is None
        
        # Test invalid JSON
        invalid_response = "not json"
        passing, reasoning, score, has_error, error_msg = await evaluator.safe_parse_eval_response(invalid_response, "correct")
        assert passing == False
        assert reasoning == "Evaluation failed due to parsing error"
        assert score == 1.0
        assert has_error == True
        assert "Failed to parse evaluation response" in error_msg
        
        # Test missing required field
        incomplete_response = '''{"score": 3, "decision": "correct"}'''  # Missing reason
        passing, reasoning, score, has_error, error_msg = await evaluator.safe_parse_eval_response(incomplete_response, "correct")
        assert passing == False
        assert reasoning == "Evaluation failed due to parsing error"
        assert score == 1.0
        assert has_error == True
        assert "Failed to parse evaluation response" in error_msg

    @pytest.mark.asyncio(loop_scope="function")
    async def test_evaluator_validation(self):
        """Test input validation in evaluate method."""
        evaluator = WandBotCorrectnessEvaluator(client=None)
        
        # Test missing query
        result = await evaluator.aevaluate(query=None, response="test", reference="test")
        assert result.has_error == True
        assert "query, response, and reference must be provided" in result.error_message
        assert result.score == 1.0
        assert result.passing == False
        
        # Test missing response
        result = await evaluator.aevaluate(query="test", response=None, reference="test")
        assert result.has_error == True
        assert "query, response, and reference must be provided" in result.error_message
        assert result.score == 1.0
        assert result.passing == False
        
        # Test missing reference
        result = await evaluator.aevaluate(query="test", response="test", reference=None)
        assert result.has_error == True
        assert "query, response, and reference must be provided" in result.error_message
        assert result.score == 1.0
        assert result.passing == False

# Integration tests with real config
@pytest.mark.integration
class TestWithRealConfig:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_evaluator_integration_real_config(self, real_config):
        """Integration test using real config and making actual OpenAI calls."""
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        evaluator = WandBotCorrectnessEvaluator(
            client=client,
            model=real_config.eval_judge_model,
            temperature=real_config.eval_judge_temperature
        )
        
        for test_case in TEST_CASES:
            result = await evaluator.aevaluate(
                query=test_case["query"],
                response=test_case["response"],
                reference=test_case["reference"],
                contexts=test_case["contexts"],
                reference_notes=test_case["reference_notes"]
            )
            
            assert isinstance(result, EvaluationResult)
            assert result.query == test_case["query"]
            assert result.response == test_case["response"]
            assert result.score == test_case["expected_score"], \
                f"Expected score {test_case['expected_score']} for query '{test_case['query']}', got {result.score}"
            assert result.passing == test_case["expected_passing"], \
                f"Expected passing={test_case['expected_passing']} for query '{test_case['query']}', got {result.passing}"
            assert result.feedback is not None and len(result.feedback) > 0
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_evaluator_edge_cases_real_config(self, real_config):
        """Test edge cases using real config and actual OpenAI calls."""
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        evaluator = WandBotCorrectnessEvaluator(
            client=client,
            model=real_config.eval_judge_model,
            temperature=real_config.eval_judge_temperature
        )
        
        # Test empty response
        result = await evaluator.aevaluate(
            query="How do I use wandb?",
            response="",
            reference="Install wandb and initialize it in your code.",
            contexts=["wandb is a tool for tracking ML experiments"]
        )
        assert result.score == 1.0
        assert result.passing == False
        
        # Test very long response
        long_response = "Use wandb. " * 100
        result = await evaluator.aevaluate(
            query="How do I use wandb?",
            response=long_response,
            reference="Install wandb and initialize it in your code.",
            contexts=["wandb is a tool for tracking ML experiments"]
        )
        assert isinstance(result.score, float)
        assert isinstance(result.passing, bool)
        
        # Test response with special characters
        special_response = "Use wandb!\n\n```python\nwandb.init()\n```\n**Note:** Important!"
        result = await evaluator.aevaluate(
            query="How do I use wandb?",
            response=special_response,
            reference="Install wandb and initialize it in your code.",
            contexts=["wandb is a tool for tracking ML experiments"]
        )
        assert isinstance(result.score, float)
        assert isinstance(result.passing, bool)
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_answer_correctness_scorer_real_config(self, real_config):
        """Test answer_correctness_scorer with real config."""
        scorer = WandbotCorrectnessScorer(config=real_config)
        
        # Test case with missing required field in model_output
        result = await scorer.score(
            question="How do I use wandb?",
            ground_truth="Install wandb and initialize it in your code.",
            notes="Basic wandb setup",
            model_output={}  # Missing generated_answer field
        )
        
        assert result["has_error"] == True
        assert result["error_message"] is not None
        assert "Generated answer is empty" in result["error_message"]
        assert result["answer_correct"] == False
        assert result["score"] == 1.0
        assert result["invalid_result"] == True
        
        # Test case with None model_output
        result = await scorer.score(
            question="How do I use wandb?",
            ground_truth="Install wandb and initialize it in your code.",
            notes="Basic wandb setup",
            model_output=None
        )
        
        assert result["has_error"] == True
        assert result["error_message"] is not None
        assert result["answer_correct"] == False
        assert result["score"] == 1.0
        assert result["invalid_result"] == True 