import os
import unittest
import warnings
from typing import List, Dict, Any, Type

import google.generativeai as genai
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel, Field

# Filter out specific protobuf warnings
warnings.filterwarnings(
    "ignore",
    message=".*PyType_Spec.*custom tp_new.*",
    category=DeprecationWarning,
    module="google._upb._message"
)

from wandbot.chat.models import (
    OpenAIChatModel,
    GeminiChatModel,
    AnthropicChatModel,
    ChatModel,
)

class ModelResponse(BaseModel):
    """Standardized model response structure that all models must follow."""
    content: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

class TestChatModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test messages that will be used across all providers
        cls.test_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Keep responses very brief."
            },
            {
                "role": "user",
                "content": "What is 2+2? Answer in one word."
            }
        ]
        
        # Define test cases for each model
        cls.model_test_cases = [
            {
                "name": "OpenAI",
                "model_class": OpenAIChatModel,
                "model_name": "gpt-4-0125-preview",
            },
            {
                "name": "Gemini",
                "model_class": GeminiChatModel,
                "model_name": "gemini-pro",
            },
            {
                "name": "Anthropic",
                "model_class": AnthropicChatModel,
                "model_name": "claude-3-opus-20240229",
            }
        ]

    def validate_response(self, response: Dict[str, Any], model_name: str):
        """Validate response structure and data types."""
        try:
            # Validate using Pydantic model
            validated_response = ModelResponse(**response)
            
            # Additional checks
            self.assertIsInstance(validated_response.content, str, 
                                f"{model_name}: content should be string")
            self.assertGreater(len(validated_response.content.strip()), 0, 
                             f"{model_name}: content should not be empty")
            
            self.assertIsInstance(validated_response.total_tokens, int, 
                                f"{model_name}: total_tokens should be integer")
            self.assertGreater(validated_response.total_tokens, 0, 
                             f"{model_name}: total_tokens should be positive")
            
            self.assertIsInstance(validated_response.prompt_tokens, int, 
                                f"{model_name}: prompt_tokens should be integer")
            self.assertGreater(validated_response.prompt_tokens, 0, 
                             f"{model_name}: prompt_tokens should be positive")
            
            self.assertIsInstance(validated_response.completion_tokens, int, 
                                f"{model_name}: completion_tokens should be integer")
            self.assertGreater(validated_response.completion_tokens, 0, 
                             f"{model_name}: completion_tokens should be positive")
            
            # Verify token counts add up
            self.assertEqual(
                validated_response.total_tokens,
                validated_response.prompt_tokens + validated_response.completion_tokens,
                f"{model_name}: total_tokens should equal prompt_tokens + completion_tokens"
            )
            
            return validated_response
            
        except Exception as e:
            self.fail(f"{model_name} response validation failed: {str(e)}")

    def test_model_interface(self):
        """Test that all models implement the required interface."""
        for test_case in self.model_test_cases:
            model_class = test_case["model_class"]
            
            # Check that model class inherits from ChatModel
            self.assertTrue(
                issubclass(model_class, ChatModel),
                f"{test_case['name']}: Model class should inherit from ChatModel"
            )
            
            # Check required methods and properties exist
            model = model_class(model_name=test_case["model_name"])
            self.assertTrue(
                hasattr(model, "generate_response"),
                f"{test_case['name']}: Model should have generate_response method"
            )
            self.assertTrue(
                hasattr(model, "system_role_key"),
                f"{test_case['name']}: Model should have system_role_key property"
            )

    def test_all_models(self):
        """Test all models with the same input and validate consistent output structure."""
        for test_case in self.model_test_cases:
            with self.subTest(model=test_case["name"]):
                print(f"\nTesting {test_case['name']} model...")
                
                # Initialize model
                model = test_case["model_class"](
                    model_name=test_case["model_name"],
                    temperature=0.1
                )
                
                # Test system_role_key property
                self.assertEqual(
                    model.system_role_key,
                    "system",
                    f"{test_case['name']}: system_role_key should be 'system'"
                )
                
                # Test generate_response
                response = model.generate_response(self.test_messages)
                
                # Validate response structure and types
                validated_response = self.validate_response(response, test_case["name"])
                
                # Print response details
                print(f"{test_case['name']} Response: {validated_response.content}")
                print(f"Token usage: {validated_response.total_tokens} total, "
                      f"{validated_response.prompt_tokens} prompt, "
                      f"{validated_response.completion_tokens} completion")

if __name__ == '__main__':
    unittest.main(verbosity=2)