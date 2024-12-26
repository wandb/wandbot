"""Tests for the ChatModel descriptor."""
import unittest
from unittest.mock import patch, MagicMock
import litellm

from wandbot.chat.chat_model import ChatModel

class TestChatModel(unittest.TestCase):
    def setUp(self):
        # Create a test class that uses the ChatModel descriptor
        class TestClass:
            chat_model = ChatModel(max_retries=2)

        self.test_obj = TestClass()
        self.test_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Test valid temperatures
        valid_temps = [0.0, 0.5, 1.0]
        for temp in valid_temps:
            self.test_obj.chat_model = {
                "model_name": "openai/gpt-4",
                "temperature": temp
            }

        # Test invalid temperatures
        invalid_temps = [-0.1, 1.1, 2.0]
        for temp in invalid_temps:
            with self.assertRaises(ValueError) as cm:
                self.test_obj.chat_model = {
                    "model_name": "openai/gpt-4",
                    "temperature": temp
                }
            self.assertIn("temperature must be between 0 and 1", str(cm.exception).lower())

    def test_message_passing(self):
        """Test that messages are passed correctly to LiteLLM."""
        test_cases = [
            # System message
            {
                "model": "openai/gpt-4",
                "messages": [{"role": "system", "content": "Be helpful"}]
            },
            # User message
            {
                "model": "anthropic/claude-3",
                "messages": [{"role": "user", "content": "Hi"}]
            },
            # Multiple messages
            {
                "model": "gemini/gemini-pro",
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hi"}
                ]
            }
        ]

        for case in test_cases:
            with patch('litellm.completion') as mock_completion:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content="Hi"))]
                mock_completion.return_value = mock_response

                # Configure model
                self.test_obj.chat_model = {
                    "model_name": case["model"],
                    "temperature": 0.1
                }

                # Call completion function
                response = self.test_obj.chat_model(case["messages"])
                
                # Verify messages are passed through unchanged
                self.assertEqual(
                    mock_completion.call_args.kwargs["messages"],
                    case["messages"]
                )

    def test_retries_and_fallbacks(self):
        """Test retry and fallback behavior."""
        # Configure model with fallbacks
        self.test_obj.chat_model = {
            "model_name": "openai/gpt-4o-mini",
            "temperature": 0.1,
            "fallback_models": ["anthropic/claude-3-haiku", "gemini/gemini-2.0-flash"]
        }

        with patch('litellm.completion') as mock_completion:
            # Mock successful response
            mock_response = MagicMock(spec=["choices", "model"])
            mock_response.choices = [MagicMock(message=MagicMock(content="Success!"))]
            mock_response.model = "anthropic/claude-3-haiku"

            # Test retry behavior
            mock_completion.side_effect = [mock_response]  # LiteLLM handles fallbacks internally

            response = self.test_obj.chat_model(self.test_messages)
            
            # Verify retry was successful
            self.assertEqual(response.choices[0].message.content, "Success!")
            self.assertEqual(response.model, "anthropic/claude-3-haiku")
            self.assertIsNone(getattr(response, "error", None))
            self.assertEqual(mock_completion.call_count, 1)  # LiteLLM handles fallbacks internally

            # Verify retry parameters were passed correctly
            call_args = mock_completion.call_args_list[0].kwargs
            self.assertEqual(call_args["num_retries"], 2)

    def test_error_handling(self):
        """Test error handling."""
        # Configure model
        self.test_obj.chat_model = {
            "model_name": "openai/gpt-4",
            "temperature": 0.1
        }

        with patch('litellm.completion') as mock_completion:
            # Test error response
            mock_completion.side_effect = Exception("Test error")
            response = self.test_obj.chat_model(self.test_messages)
            
            # Verify error response
            self.assertEqual(response.choices[0].message.content, "")
            self.assertEqual(response.error["type"], "Exception")
            self.assertEqual(response.error["message"], "Test error")
            self.assertEqual(response.model, "openai/gpt-4")

if __name__ == '__main__':
    unittest.main()