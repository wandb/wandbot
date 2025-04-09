import os

import pytest
from openai import AsyncOpenAI

from wandbot.evaluation.eval_metrics.correctness import WandBotCorrectnessEvaluator


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that make real API calls to external services"
    )

@pytest.fixture(scope="function")
def evaluator():
    """Create an evaluator instance for testing."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return WandBotCorrectnessEvaluator(client=client)

def pytest_collection_modifyitems(config, items):
    """Modify test items in-place to ensure proper async test behavior."""
    for item in items:
        # Add asyncio marker to all async tests
        if "async" in item.keywords:
            item.add_marker(pytest.mark.asyncio)
        
        # Skip integration tests if SKIP_INTEGRATION_TESTS is set
        if "integration" in item.keywords and os.getenv("SKIP_INTEGRATION_TESTS"):
            item.add_marker(pytest.mark.skip(reason="Integration tests are disabled")) 