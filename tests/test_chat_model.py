"""Tests for the ChatModel class."""
import unittest
from unittest.mock import patch, MagicMock, call
import litellm

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
            mock_completion.call_args = MagicMock(
                kwargs={
                    "model": "openai/gpt-4",
                    "messages": [
                        {"role": "developer", "content": "You are a helpful assistant"},
                        {"role": "user", "content": "Hello"}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "timeout": None,
                    "num_retries": 3
                }
            )
            
            response = self.model.generate_response(self.test_messages)
            
            # Verify system role was converted to developer
            self.assertEqual(
                mock_completion.call_args.kwargs["messages"][0]["role"],
                "developer"
            )
            self.assertEqual(
                mock_completion.call_args.kwargs["messages"][1]["role"],
                "user"
            )

    def test_error_handling(self):
        """Test error handling."""
        with patch('litellm.completion') as mock_completion:
            # Test retryable error
            mock_completion.side_effect = litellm.exceptions.RateLimitError(
                message="Rate limit exceeded",
                llm_provider="openai",
                model="gpt-4"
            )
            response = self.model.generate_response(self.test_messages)
            self.assertTrue(response["error"]["retryable"])

            # Test non-retryable error
            mock_completion.side_effect = litellm.exceptions.AuthenticationError(
                message="Invalid API key",
                llm_provider="openai",
                model="gpt-4"
            )
            response = self.model.generate_response(self.test_messages)
            self.assertFalse(response["error"]["retryable"])

            # Test server error
            mock_completion.side_effect = litellm.exceptions.ServiceUnavailableError(
                message="Internal server error",
                llm_provider="openai",
                model="gpt-4"
            )
            response = self.model.generate_response(self.test_messages)
            self.assertTrue(response["error"]["retryable"])

    def test_message_format_validation(self):
        """Test message format validation."""
        invalid_messages = [
            # Missing role
            [{"content": "Hello"}],
            # Missing content
            [{"role": "user"}],
            # Invalid role
            [{"role": "invalid", "content": "Hello"}],
            # Wrong types
            [{"role": 123, "content": "Hello"}],
            [{"role": "user", "content": 123}],
            # Empty messages
            [],
            # None messages
            None,
        ]

        for messages in invalid_messages:
            with patch('litellm.completion') as mock_completion:
                response = self.model.generate_response(messages)
                self.assertIsNotNone(response["error"])
                self.assertEqual(response["error"]["type"], "ValueError")
                self.assertFalse(response["error"]["retryable"])

    def test_different_providers(self):
        """Test different model providers handle their quirks."""
        test_cases = [
            # OpenAI uses "developer" role
            {
                "model": "openai/gpt-4",
                "messages": [{"role": "system", "content": "Be helpful"}],
                "expected_role": "developer"
            },
            # Anthropic handles system message separately
            {
                "model": "anthropic/claude-3",
                "messages": [{"role": "system", "content": "Be helpful"}],
                "expected_system": True
            },
            # Gemini prepends system to first user message
            {
                "model": "gemini/gemini-pro",
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hi"}
                ],
                "expected_prepend": True
            }
        ]

        for case in test_cases:
            model = ChatModel(model_name=case["model"])
            with patch('litellm.completion') as mock_completion:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content="Hi"))]
                mock_response.usage = MagicMock(
                    total_tokens=10,
                    prompt_tokens=5,
                    completion_tokens=5
                )
                mock_completion.return_value = mock_response
                mock_completion.call_args = MagicMock(
                    kwargs={
                        "model": case["model"],
                        "messages": case["messages"],
                        "max_tokens": 1000,
                        "temperature": 0.1,
                        "timeout": None,
                        "num_retries": 3
                    }
                )

                response = model.generate_response(case["messages"])
                
                # Verify provider-specific handling
                messages = mock_completion.call_args.kwargs["messages"]
                if "expected_role" in case:
                    self.assertEqual(messages[0]["role"], case["expected_role"])
                if "expected_system" in case:
                    self.assertEqual(messages[0]["role"], "system")
                if "expected_prepend" in case:
                    self.assertTrue(
                        case["messages"][0]["content"] in messages[0]["content"]
                    )

    def test_retries_and_fallbacks(self):
        """Test retry and fallback behavior."""
        model = ChatModel(
            model_name="openai/gpt-4o-mini",
            fallback_models=["anthropic/claude-3-haiku", "gemini/gemini-2.0-flash"],
            num_retries=2,
            timeout=10,
            max_backoff=5
        )

        with patch('litellm.completion') as mock_completion:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Success!"))]
            mock_response.usage = MagicMock(
                total_tokens=10,
                prompt_tokens=5,
                completion_tokens=5
            )
            mock_response.model = "anthropic/claude-3-haiku"

            # Test retry behavior
            mock_completion.side_effect = [mock_response]  # LiteLLM handles fallbacks internally

            response = model.generate_response(self.test_messages)
            
            # Verify retry was successful
            self.assertEqual(response["content"], "Success!")
            self.assertEqual(response["model_used"], "anthropic/claude-3-haiku")
            self.assertIsNone(response["error"])
            self.assertEqual(mock_completion.call_count, 1)  # LiteLLM handles fallbacks internally

            # Verify retry parameters were passed correctly
            call_args = mock_completion.call_args_list[0].kwargs
            self.assertEqual(call_args["num_retries"], 2)
            self.assertEqual(call_args["timeout"], 10)

    def test_context_window_limits(self):
        """Test handling of context window limits with modern models."""
        # Create a massive message (5M tokens)
        # Each word is roughly 1.3 tokens, so we need about 3.8M words
        long_content = "test " * 3_800_000
        messages = [{"role": "user", "content": long_content}]

        test_cases = [
            # OpenAI GPT-4 Turbo Mini (128K tokens)
            ("openai/gpt-4o-mini", 128_000),
            # Gemini 2.0 Flash (1M tokens)
            ("gemini/gemini-2.0-flash", 1_000_000),
            # Claude 3 Haiku (200K tokens)
            ("anthropic/claude-3-haiku", 200_000),
        ]

        for model_name, context_limit in test_cases:
            model = ChatModel(model_name=model_name)
            with patch('litellm.completion') as mock_completion:
                # Each provider has slightly different error messages
                if "openai" in model_name:
                    error = litellm.exceptions.ContextWindowExceededError(
                        message=f"This model's maximum context length is {context_limit} tokens. "
                               f"However, your messages resulted in {5_000_000} tokens",
                        llm_provider="openai",
                        model=model_name
                    )
                elif "anthropic" in model_name:
                    error = litellm.exceptions.ContextWindowExceededError(
                        message=f"This content exceeds the maximum length of {context_limit} tokens",
                        llm_provider="anthropic",
                        model=model_name
                    )
                else:  # gemini
                    error = litellm.exceptions.ContextWindowExceededError(
                        message=f"Input length of {5_000_000} tokens exceeds maximum context length of {context_limit} tokens",
                        llm_provider="google",
                        model=model_name
                    )

                mock_completion.side_effect = error
                response = model.generate_response(messages)
                
                # Verify error response
                self.assertIsNotNone(response["error"])
                self.assertFalse(response["error"]["retryable"])
                self.assertEqual(response["model_used"], model_name)
                
                # Verify error message contains the limit
                error_msg = response["error"]["message"].lower()
                self.assertIn(str(context_limit), error_msg)
                self.assertIn("token", error_msg)
                self.assertTrue(
                    any(word in error_msg for word in ["context", "length", "exceed"])
                )

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Test valid temperatures
        valid_temps = [0.0, 0.5, 1.0]
        for temp in valid_temps:
            model = ChatModel(model_name="openai/gpt-4", temperature=temp)
            self.assertEqual(model.temperature, temp)

        # Test invalid temperatures
        invalid_temps = [-0.1, 1.1, 2.0]
        for temp in invalid_temps:
            with self.assertRaises(ValueError) as cm:
                ChatModel(model_name="openai/gpt-4", temperature=temp)
            self.assertIn("temperature must be between 0 and 1", str(cm.exception).lower())

if __name__ == '__main__':
    unittest.main()