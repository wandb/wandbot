import unittest
from unittest.mock import patch, MagicMock
import anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import Message, MessageParam, ContentBlock

from wandbot.chat.models import AnthropicChatModel
from wandbot.chat.models.base import ModelError

class TestAnthropicModel(unittest.TestCase):
    def setUp(self):
        self.model = AnthropicChatModel()

    def test_system_message_handling(self):
        """Test that system message is passed correctly to Anthropic API."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello! How can I help?")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        with patch.object(self.model.client.messages, 'create') as mock_create:
            mock_create.return_value = mock_response
            
            self.model.generate_response(messages)
            
            # Verify system message was passed correctly
            call_args = mock_create.call_args[1]
            self.assertEqual(call_args["system"], "You are a helpful assistant")
            
            # Verify messages were formatted correctly
            self.assertEqual(len(call_args["messages"]), 1)  # System message not included in messages
            self.assertEqual(call_args["messages"][0]["role"], "user")
            self.assertEqual(call_args["messages"][0]["content"][0]["text"], "Hello")

    def test_no_system_message(self):
        """Test handling when no system message is present."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hi!")]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 2

        with patch.object(self.model.client.messages, 'create') as mock_create:
            mock_create.return_value = mock_response
            
            self.model.generate_response(messages)
            
            # Verify system parameter was not included
            call_args = mock_create.call_args[1]
            self.assertEqual(call_args.get("system"), NOT_GIVEN)

    def test_message_format(self):
        """Test that messages are formatted correctly for Anthropic API."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "user", "content": "Another user message"}
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 5

        with patch.object(self.model.client.messages, 'create') as mock_create:
            mock_create.return_value = mock_response
            
            self.model.generate_response(messages)
            
            # Verify message format
            call_args = mock_create.call_args[1]
            anthropic_messages = call_args["messages"]
            
            self.assertEqual(len(anthropic_messages), 3)  # System message handled separately
            
            # Check message structure
            self.assertEqual(anthropic_messages[0]["role"], "user")
            self.assertEqual(anthropic_messages[0]["content"][0]["type"], "text")
            self.assertEqual(anthropic_messages[0]["content"][0]["text"], "User message")
            
            self.assertEqual(anthropic_messages[1]["role"], "assistant")
            self.assertEqual(anthropic_messages[1]["content"][0]["text"], "Assistant message")
            
            self.assertEqual(anthropic_messages[2]["role"], "user")
            self.assertEqual(anthropic_messages[2]["content"][0]["text"], "Another user message")

    def test_error_handling(self):
        """Test handling of Anthropic-specific errors."""
        messages = [{"role": "user", "content": "Hello"}]

        # Test invalid input error
        response = self.model.generate_response([])  # Empty messages list
        self.assertEqual(response["content"], "")
        self.assertIsInstance(response["error"], ModelError)
        self.assertEqual(response["error"].type, "invalid_input")
        self.assertFalse(response["error"].retryable)

        # Test connection error
        with patch.object(self.model.client.messages, 'create') as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            response = self.model.generate_response(messages)
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "unknown_error")
            self.assertTrue(response["error"].retryable)