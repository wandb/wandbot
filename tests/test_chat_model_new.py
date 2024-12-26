import unittest
from unittest.mock import patch, MagicMock
import litellm

from wandbot.chat.models import GeminiChatModel
from wandbot.chat.models.base import ModelError

class TestChatModel(unittest.TestCase):
    def setUp(self):
        self.model = GeminiChatModel(
            model_name="openai/gpt-4",
            fallback_models=["anthropic/claude-3", "gemini/gemini-pro"]
        )
        self.test_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]

    def test_openai_role_conversion(self):
        """Test that system role is converted to developer for OpenAI models."""
        with patch('litellm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Hi!"))
            ]
            mock_response.usage = MagicMock(
                total_tokens=10,
                prompt_tokens=8,
                completion_tokens=2
            )
            mock_response.model = "openai/gpt-4"
            mock_completion.return_value = mock_response
            
            self.model.generate_response(self.test_messages)
            
            # Verify system role was converted to developer
            call_args = mock_completion.call_args[1]
            self.assertEqual(call_args["messages"][0]["role"], "developer")
            self.assertEqual(call_args["messages"][1]["role"], "user")

    def test_error_handling(self):
        """Test handling of various error types."""
        error_cases = [
            (litellm.exceptions.RateLimitError("Rate limit exceeded"), "rate_limit", True),
            (litellm.exceptions.InvalidRequestError("Invalid request"), "invalid_request", False),
            (litellm.exceptions.AuthenticationError("Invalid key"), "auth_error", False),
            (litellm.exceptions.APIConnectionError("Connection failed"), "connection_error", True),
            (litellm.exceptions.ContextLengthExceededError("Too long"), "context_length", False),
            (litellm.exceptions.ServiceUnavailableError("Service down"), "service_unavailable", True),
        ]

        for error, expected_type, retryable in error_cases:
            with patch('litellm.completion') as mock_completion:
                mock_completion.side_effect = error
                
                response = self.model.generate_response(self.test_messages)
                
                # Verify error response format
                self.assertEqual(response["content"], "")
                self.assertEqual(response["total_tokens"], 0)
                self.assertEqual(response["prompt_tokens"], 0)
                self.assertEqual(response["completion_tokens"], 0)
                self.assertEqual(response["model_used"], "openai/gpt-4")
                
                # Verify error details
                self.assertIsNotNone(response["error"])
                self.assertEqual(response["error"]["type"], expected_type)
                self.assertEqual(response["error"]["retryable"], retryable)

    def test_model_fallback(self):
        """Test fallback to backup model on error."""
        with patch('litellm.completion') as mock_completion:
            # Mock successful response from fallback model
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Fallback response"))
            ]
            mock_response.usage = MagicMock(
                total_tokens=5,
                prompt_tokens=3,
                completion_tokens=2
            )
            mock_response.model = "anthropic/claude-3"
            
            # Configure mock to fail first call and succeed second call
            mock_completion.side_effect = [
                litellm.exceptions.RateLimitError("Rate limit exceeded"),  # Primary model fails
                mock_response  # Fallback model succeeds
            ]
            
            response = self.model.generate_response(self.test_messages)
            
            # Verify fallback was successful
            self.assertEqual(response["content"], "Fallback response")
            self.assertEqual(response["model_used"], "anthropic/claude-3")
            self.assertIsNone(response["error"])
            self.assertEqual(response["total_tokens"], 5)

    def test_successful_response(self):
        """Test successful response handling."""
        with patch('litellm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Hello! How can I help?"))
            ]
            mock_response.usage = MagicMock(
                total_tokens=15,
                prompt_tokens=10,
                completion_tokens=5
            )
            mock_response.model = "openai/gpt-4"
            mock_completion.return_value = mock_response
            
            response = self.model.generate_response(self.test_messages)
            
            # Verify response format
            self.assertEqual(response["content"], "Hello! How can I help?")
            self.assertEqual(response["total_tokens"], 15)
            self.assertEqual(response["prompt_tokens"], 10)
            self.assertEqual(response["completion_tokens"], 5)
            self.assertEqual(response["model_used"], "openai/gpt-4")
            self.assertIsNone(response["error"])

if __name__ == '__main__':
    unittest.main()