import sys
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from wandbot.evaluation.eval_config import EvalConfig
from wandbot.evaluation.eval_metrics.correctness import (
    SYSTEM_TEMPLATE,
    USER_TEMPLATE,
    CorrectnessEvaluationModel,
    CorrectnessEvaluationResult,
    WandBotCorrectnessEvaluator,
)
from wandbot.models.llm import LLMError
from wandbot.schema.api_status import APIStatus, ErrorInfo


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
        "response": "To log metrics in wandb, use wandb.log() with a dictionary of metrics. For example: wandb.log({'loss': 0.2, 'accuracy': 0.85}). This will log your metrics to the W&B dashboard where you can visualize them in real-time.",
        "reference": "Use wandb.log() to log metrics. Pass a dictionary with metric names as keys and values as the metric values. Example: wandb.log({'loss': 0.2, 'accuracy': 0.85})",
        "reference_notes": "This is the correct way to log metrics in wandb using the wandb.log() method.",
        "contexts": ["wandb.log() is used to log metrics during training. It accepts a dictionary of metric names and values."],
        "expected_score": 3.0,
        "expected_passing": True,
        "reason": "Perfect match with reference - provides complete, accurate information with example"
    },
    {
        "query": "How do I save a model checkpoint?",
        "response": "Use TensorFlow's model.save() function to save your model.",
        "reference": "To save model checkpoints with wandb, use either wandb.save() to save files directly, or wandb.log_artifact() to version and track your models.",
        "reference_notes": "The response should mention wandb-specific methods for saving checkpoints.",
        "contexts": ["Model checkpoints can be saved using wandb.save() or wandb.log_artifact()"],
        "expected_score": 1.0,
        "expected_passing": False,
        "reason": "Completely incorrect - suggests non-wandb method when wandb-specific methods exist"
    },
    {
        "query": "How do I log images in wandb?",
        "response": "Use wandb.log() with wandb.Image. Example: wandb.log({'image': wandb.Image(array)})",
        "reference": "To log images, use wandb.log() with wandb.Image. You can log PNG, JPG, or other image formats. You can also add optional captions and masks. Example: wandb.log({'image': wandb.Image(img_array, caption='My Image')})",
        "reference_notes": "The response should mention wandb.log with wandb.Image and ideally include format support and optional parameters.",
        "contexts": ["Images can be logged using wandb.log() with wandb.Image objects. Supports various formats and optional parameters like captions."],
        "expected_score": 2.0,
        "expected_passing": False,
        "reason": "Partially correct - has basic functionality but missing important details about formats and options"
    }
]

# Basic unit tests with mock config
@pytest.mark.usefixtures("mock_config")
class TestWithMockConfig:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_safe_parse_eval_response(self):
        """Test parsing of evaluation responses."""
        evaluator = WandBotCorrectnessEvaluator()
        
        # Mock the LLM response
        mock_response = CorrectnessEvaluationModel(
            reason="test",
            score=3.0,
            decision="correct"
        )
        api_status = APIStatus(component="llm", success=True)
        evaluator.llm.create = AsyncMock(return_value=(mock_response, api_status))
        
        # Test with valid prompts
        result = await evaluator._get_completion(
            system_prompt=SYSTEM_TEMPLATE,
            user_prompt=USER_TEMPLATE.format(
                query="test query",
                generated_answer="test response",
                reference_answer="test reference",
                context_str="test context",
                reference_notes="test notes"
            )
        )
        assert isinstance(result, CorrectnessEvaluationModel)
        assert result.score == 3.0
        assert result.decision == "correct"
        assert result.reason == "test"

        # Test error case
        error_status = APIStatus(component="llm", success=False, error_info=ErrorInfo(has_error=True, error_message="test error"))
        evaluator.llm.create = AsyncMock(return_value=(None, error_status))
        
        result = await evaluator._get_completion(
            system_prompt=SYSTEM_TEMPLATE,
            user_prompt=USER_TEMPLATE.format(
                query="test query",
                generated_answer="test response",
                reference_answer="test reference",
                context_str="test context",
                reference_notes="test notes"
            )
        )
        assert isinstance(result, LLMError)
        assert result.error
        assert result.error_message == "test error"

        # Test full evaluation with error
        result = await evaluator.aevaluate(
            query="test",
            response="test",
            reference="test",
            contexts=["test context"]
        )
        assert isinstance(result, CorrectnessEvaluationResult)
        assert result.score == 1.0
        assert result.decision == "incorrect"
        assert result.has_error
        assert result.error_message == "test error"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_evaluator_validation(self):
        """Test input validation in evaluate method."""
        evaluator = WandBotCorrectnessEvaluator()
        
        # Mock successful LLM response
        mock_response = CorrectnessEvaluationModel(
            reason="test",
            score=3.0,
            decision="correct"
        )
        api_status = APIStatus(component="llm", success=True)
        evaluator.llm.create = AsyncMock(return_value=(mock_response, api_status))
        
        # Test missing query
        result = await evaluator.aevaluate(
            query=None,
            response="test",
            reference="test",
            contexts=["test context"]
        )
        assert result.has_error
        assert "query, response, and reference must be provided" in result.error_message
        assert result.score == 1.0
        assert result.decision == "incorrect"
        
        # Test missing response
        result = await evaluator.aevaluate(
            query="test",
            response=None,
            reference="test",
            contexts=["test context"]
        )
        assert result.has_error
        assert "query, response, and reference must be provided" in result.error_message
        assert result.score == 1.0
        assert result.decision == "incorrect"
        
        # Test missing reference
        result = await evaluator.aevaluate(
            query="test",
            response="test",
            reference=None,
            contexts=["test context"]
        )
        assert result.has_error
        assert "query, response, and reference must be provided" in result.error_message
        assert result.score == 1.0
        assert result.decision == "incorrect"

# Integration tests with real config
@pytest.mark.integration
class TestWithRealConfig:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_evaluator_integration_real_config(self, real_config):
        """Integration test using real config and making actual OpenAI calls."""
        evaluator = WandBotCorrectnessEvaluator(
            model_name=real_config.eval_judge_model,
            temperature=real_config.eval_judge_temperature,
            provider="openai"  # Add provider explicitly
        )
        
        for test_case in TEST_CASES:
            result = await evaluator.aevaluate(
                query=test_case["query"],
                response=test_case["response"],
                reference=test_case["reference"],
                contexts=test_case["contexts"],
                reference_notes=test_case["reference_notes"]
            )
            
            assert isinstance(result, CorrectnessEvaluationResult)
            assert result.score == test_case["expected_score"], \
                f"Expected score {test_case['expected_score']} for query '{test_case['query']}', got {result.score}"
            assert (result.decision == "correct") == test_case["expected_passing"], \
                f"Expected passing={test_case['expected_passing']} for query '{test_case['query']}', got {result.decision}"
            assert result.reason is not None and len(result.reason) > 0
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_evaluator_edge_cases_real_config(self, real_config):
        """Test edge cases using real config and actual OpenAI calls."""
        evaluator = WandBotCorrectnessEvaluator(
            model_name=real_config.eval_judge_model,
            temperature=real_config.eval_judge_temperature,
            provider="openai"  # Add provider explicitly
        )
        
        # Test empty response
        result = await evaluator.aevaluate(
            query="How do I use wandb?",
            response="",
            reference="Install wandb and initialize it in your code.",
            contexts=["wandb is a tool for tracking ML experiments"]
        )
        assert result.score == 1.0
        assert result.decision == "incorrect"
        
        # Test very long response
        long_response = "Use wandb. " * 100
        result = await evaluator.aevaluate(
            query="How do I use wandb?",
            response=long_response,
            reference="Install wandb and initialize it in your code.",
            contexts=["wandb is a tool for tracking ML experiments"]
        )
        assert isinstance(result.score, float)
        assert isinstance(result.decision, str)
        
        # Test response with special characters
        special_response = "Use wandb!\n\n```python\nwandb.init()\n```\n**Note:** Important!"
        result = await evaluator.aevaluate(
            query="How do I use wandb?",
            response=special_response,
            reference="Install wandb and initialize it in your code.",
            contexts=["wandb is a tool for tracking ML experiments"]
        )
        assert isinstance(result.score, float)
        assert isinstance(result.decision, str)
        assert result.reason is not None 