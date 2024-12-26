import unittest
from unittest.mock import patch, MagicMock
from google.api_core import exceptions as google_exceptions

from wandbot.chat.models import GeminiChatModel
from wandbot.chat.models.base import ModelError

class TestGeminiModelErrors(unittest.TestCase):
    def setUp(self):
        self.model = GeminiChatModel(
            model_name="gemini-pro",
            fallback_model="gemini-1.0-pro",  # Specify a fallback model
            fallback_temperature=0.2
        )
        self.test_messages = [
            {"role": "user", "content": "Hello"}
        ]

    def test_auth_error(self):
        """Test handling of authentication errors."""
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat.side_effect = google_exceptions.PermissionDenied("Invalid API key")
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "")
            self.assertEqual(response["total_tokens"], 0)
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "auth_error")
            self.assertFalse(response["error"].retryable)

    def test_rate_limit_error(self):
        """Test handling of rate limit errors."""
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat.side_effect = google_exceptions.ResourceExhausted("Rate limit exceeded")
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "rate_limit")
            self.assertTrue(response["error"].retryable)

    def test_context_length_error(self):
        """Test handling of context length errors."""
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat.side_effect = google_exceptions.InvalidArgument(
                "Input too long"
            )
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "invalid_request")
            self.assertFalse(response["error"].retryable)

    def test_server_error(self):
        """Test handling of server errors."""
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat.side_effect = google_exceptions.InternalServerError("Server error")
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "server_error")
            self.assertTrue(response["error"].retryable)

    def test_safety_error(self):
        """Test handling of safety-related errors."""
        mock_response = MagicMock()
        mock_response.prompt_feedback = {"safety": "blocked"}
        
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat_instance = MagicMock()
            mock_chat_instance.send_message.return_value = mock_response
            mock_chat.return_value = mock_chat_instance
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "safety_error")
            self.assertEqual(response["error"].code, "SAFETY_BLOCK")
            self.assertFalse(response["error"].retryable)

    def test_invalid_input(self):
        """Test handling of invalid input."""
        response = self.model.generate_response([])  # Empty messages list
        
        self.assertEqual(response["content"], "")
        self.assertIsInstance(response["error"], ModelError)
        self.assertEqual(response["error"].type, "invalid_input")
        self.assertFalse(response["error"].retryable)

    def test_successful_response_has_no_error(self):
        """Test that successful responses have error=None."""
        mock_response = MagicMock()
        mock_response.text = "Hello!"
        mock_response.prompt_feedback = None
        mock_response.usage_metadata = MagicMock(
            total_token_count=10,
            prompt_token_count=5,
            candidates_token_count=5
        )
        
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat_instance = MagicMock()
            mock_chat_instance.send_message.return_value = mock_response
            mock_chat.return_value = mock_chat_instance
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "Hello!")
            self.assertIsNone(response["error"])
            self.assertEqual(response["total_tokens"], 10)
            self.assertEqual(response["prompt_tokens"], 5)
            self.assertEqual(response["completion_tokens"], 5)

    def test_network_timeout(self):
        """Test handling of network timeout errors."""
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat.side_effect = TimeoutError("Request timed out")
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "timeout")
            self.assertTrue(response["error"].retryable)

    def test_network_connectivity(self):
        """Test handling of network connectivity errors."""
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat.side_effect = ConnectionError("Failed to connect")
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "network_error")
            self.assertTrue(response["error"].retryable)

    def test_model_not_found(self):
        """Test handling of non-existent model."""
        with patch('google.generativeai.GenerativeModel.__init__') as mock_init:
            mock_init.side_effect = google_exceptions.NotFound("Model not found")
            
            with self.assertRaises(RuntimeError) as context:
                GeminiChatModel(model_name="non-existent-model")
            
            self.assertIn("Failed to initialize", str(context.exception))
            self.assertIn("Model not found", str(context.exception))

    def test_model_not_ready(self):
        """Test handling of model not ready errors."""
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            mock_chat.side_effect = google_exceptions.FailedPrecondition("Model not ready")
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "")
            self.assertIsInstance(response["error"], ModelError)
            self.assertEqual(response["error"].type, "model_error")
            self.assertTrue(response["error"].retryable)

    def test_invalid_temperature(self):
        """Test handling of invalid temperature parameter."""
        with self.assertRaises(ValueError):
            GeminiChatModel(temperature=2.0)  # Temperature should be between 0 and 1

    def test_empty_messages(self):
        """Test handling of empty messages list."""
        response = self.model.generate_response([])
        self.assertEqual(response["content"], "")
        self.assertIsInstance(response["error"], ModelError)
        self.assertEqual(response["error"].type, "invalid_input")

    def test_invalid_message_role(self):
        """Test handling of invalid message role."""
        messages = [{"role": "invalid_role", "content": "Hello"}]
        response = self.model.generate_response(messages)
        self.assertEqual(response["content"], "")
        self.assertIsInstance(response["error"], ModelError)
        self.assertEqual(response["error"].type, "invalid_input")

    def test_fallback_on_error(self):
        """Test fallback behavior when primary model fails."""
        # Mock primary model to fail
        mock_response = MagicMock()
        mock_response.text = "Fallback response"
        mock_response.prompt_feedback = None
        mock_response.usage_metadata = MagicMock(
            total_token_count=5,
            prompt_token_count=2,
            candidates_token_count=3
        )

        with patch('google.generativeai.GenerativeModel.start_chat') as mock_chat:
            # First call fails, second call (fallback) succeeds
            mock_chat_instance = MagicMock()
            mock_chat_instance.send_message.side_effect = [
                google_exceptions.InternalServerError("Primary model failed"),
                mock_response
            ]
            mock_chat.return_value = mock_chat_instance
            
            response = self.model.generate_response(self.test_messages)
            
            self.assertEqual(response["content"], "Fallback response")
            self.assertIsNone(response["error"])
            self.assertEqual(response["total_tokens"], 5)

if __name__ == '__main__':
    unittest.main()