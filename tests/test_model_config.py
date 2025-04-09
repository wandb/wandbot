"""Test configurations for language models."""

from typing import List

# Available Models for Testing
ANTHROPIC_MODELS: List[str] = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022"
]

OPENAI_MODELS: List[str] = [
    "gpt-4-1106-preview",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "o1-2024-12-17",
    # "o1-mini-2024-09-12",
    "o3-mini-2025-01-31"
]

# Default configurations for each provider in tests
MODEL_CONFIGS = {
    "anthropic": {
        "provider": "anthropic",
        "temperature": 0.7,
    },
    "openai": {
        "provider": "openai",
        "temperature": 0.7,
    }
}

# Test configurations
TEST_CONFIG = {
    "primary": {
        "provider": "anthropic",
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
    },
    "fallback": {
        "provider": "openai", 
        "model_name": "gpt-4-1106-preview",
        "temperature": 0.7,
    }
}

# Invalid model configurations for testing fallback behavior
TEST_INVALID_MODELS = {
    "primary": {
        "provider": "anthropic",
        "model_name": "invalid-model-1",
        "temperature": 0.7,
    },
    "fallback": {
        "provider": "openai",
        "model_name": "invalid-model-2", 
        "temperature": 0.7,
    }
} 