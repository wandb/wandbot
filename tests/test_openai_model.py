import unittest
from unittest.mock import patch, MagicMock
from openai import OpenAIError, APIError, RateLimitError

from wandbot.chat.models import OpenAIChatModel
from wandbot.chat.models.base import ModelError

class TestOpenAIModel(unittest.TestCase):
    def setUp(self):
        self.model = OpenAIChatModel()

    def test_system_role_conversion(self):
        """Test that system role is converted to developer role."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hi!"))]
        mock_response.usage.total_tokens = 10
        mock_response.usage.prompt_tokens = 8
        mock_response.usage.completion_tokens = 2

        with patch.object(self.model.client.chat.completions, 'create') as mock_create:
            mock_create.return_value = mock_response
            
            self.model.generate_response(messages)
            
            # Verify system role was converted to developer
            call_args = mock_create.call_args[1]
            self.assertEqual(len(call_args["messages"]), 2)
            self.assertEqual(call_args["messages"][0]["role"], "developer")
            self.assertEqual(call_args["messages"][1]["role"], "user")

    def test_error_handling(self):
        """Test handling of OpenAI-specific errors."""
        messages = [{"role": "user", "content": "Hello"}]

        # Test generic error
        with patch.object(self.model.client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = OpenAIError("OpenAI error")
            response = self.model.generate_response(messages)
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "api_error")
            self.assertTrue(response["error"].retryable)

        # Test connection error
        with patch.object(self.model.client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            response = self.model.generate_response(messages)
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "unknown_error")
            self.assertTrue(response["error"].retryable)

    def test_input_validation(self):
        """Test input validation."""
        # Test empty messages
        response = self.model.generate_response([])
        self.assertEqual(response["content"], "")
        self.assertIsInstance(response["error"], ModelError)
        self.assertEqual(response["error"].type, "invalid_input")
        self.assertFalse(response["error"].retryable)

        # Test invalid role
        response = self.model.generate_response([
            {"role": "invalid_role", "content": "Hello"}
        ])
        self.assertEqual(response["content"], "")
        self.assertIsInstance(response["error"], ModelError)
        self.assertEqual(response["error"].type, "invalid_input")
        self.assertFalse(response["error"].retryable)

        # Test missing content
        response = self.model.generate_response([
            {"role": "user"}
        ])
        self.assertEqual(response["content"], "")
        self.assertIsInstance(response["error"], ModelError)
        self.assertEqual(response["error"].type, "invalid_input")
        self.assertFalse(response["error"].retryable)

    def test_successful_response(self):
        """Test successful response handling."""
        messages = [{"role": "user", "content": "Hello"}]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hi!"))]
        mock_response.usage.total_tokens = 5
        mock_response.usage.prompt_tokens = 3
        mock_response.usage.completion_tokens = 2

        with patch.object(self.model.client.chat.completions, 'create') as mock_create:
            mock_create.return_value = mock_response
            
            response = self.model.generate_response(messages)
            
            self.assertEqual(response["content"], "Hi!")
            self.assertIsNone(response["error"])
            self.assertEqual(response["total_tokens"], 5)
            self.assertEqual(response["prompt_tokens"], 3)
            self.assertEqual(response["completion_tokens"], 2)