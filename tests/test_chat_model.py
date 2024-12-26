import unittest
from unittest.mock import patch, MagicMock

from wandbot.chat.chat_model import ChatModel

class TestChatModel(unittest.TestCase):
    def setUp(self):
        self.model = ChatModel(model_name="openai/gpt-4")
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
        """Test error handling."""
        with patch('litellm.completion') as mock_completion:
            # Test retryable error
            mock_completion.side_effect = Exception("Rate limit exceeded")
            response = self.model.generate_response(self.test_messages)
            self.assertTrue(response["error"]["retryable"])

            # Test non-retryable error
            mock_completion.side_effect = Exception("Invalid API key")
            response = self.model.generate_response(self.test_messages)
            self.assertFalse(response["error"]["retryable"])

            # Test server error
            mock_completion.side_effect = Exception("Internal server error")
            response = self.model.generate_response(self.test_messages)
            self.assertTrue(response["error"]["retryable"])

            # Verify error response format
            self.assertEqual(response["content"], "")
            self.assertEqual(response["total_tokens"], 0)
            self.assertEqual(response["prompt_tokens"], 0)
            self.assertEqual(response["completion_tokens"], 0)
            self.assertEqual(response["model_used"], "openai/gpt-4")
            self.assertIsNotNone(response["error"])

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