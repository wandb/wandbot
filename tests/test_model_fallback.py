import unittest
from unittest.mock import patch, MagicMock
from google.api_core import exceptions as google_exceptions

from wandbot.chat.models import GeminiChatModel
from mock_models import MockOpenAIModel
from wandbot.chat.models.base import ModelError

class TestModelFallback(unittest.TestCase):
    def setUp(self):
        # Create a Gemini model with OpenAI fallback
        self.fallback_model = MockOpenAIModel("gpt-4")
        self.primary_model = GeminiChatModel(
            model_name="gemini-pro",
            fallback_model=self.fallback_model
        )
        self.test_messages = [
            {"role": "user", "content": "Hello"}
        ]

    def test_fallback_on_retryable_error(self):
        """Test fallback to OpenAI when Gemini has a retryable error."""
        # Mock Gemini to fail with a retryable error
        mock_gemini_error = google_exceptions.InternalServerError("Gemini server error")
        
        # Mock OpenAI to succeed
        mock_openai_response = {
            "content": "Fallback response from OpenAI",
            "total_tokens": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "error": None,
            "model_used": "gpt-4"
        }
        
        # Setup mocks
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_gemini:
            mock_gemini.side_effect = mock_gemini_error
            
            with patch.object(self.fallback_model, 'generate_response') as mock_openai:
                mock_openai.return_value = mock_openai_response
                
                # Make the request
                response = self.primary_model.generate_response(self.test_messages)
                
                # Verify fallback was used
                self.assertEqual(response["content"], "Fallback response from OpenAI")
                self.assertEqual(response["model_used"], "gpt-4")
                self.assertIsNone(response["error"])

    def test_no_fallback_on_non_retryable_error(self):
        """Test that non-retryable errors don't trigger fallback."""
        # Mock Gemini to fail with a non-retryable error
        mock_gemini_error = google_exceptions.PermissionDenied("Invalid API key")
        
        # Setup mocks
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_gemini:
            mock_gemini.side_effect = mock_gemini_error
            
            with patch.object(self.fallback_model, 'generate_response') as mock_openai:
                # Make the request
                response = self.primary_model.generate_response(self.test_messages)
                
                # Verify fallback was not used
                self.assertEqual(response["content"], "")
                self.assertEqual(response["model_used"], "gemini-pro")
                self.assertEqual(response["error"].type, "auth_error")
                mock_openai.assert_not_called()

    def test_fallback_chain(self):
        """Test chaining multiple fallbacks."""
        # Create a chain of fallbacks
        final_fallback = MockOpenAIModel("gpt-3.5-turbo")
        middle_fallback = GeminiChatModel(
            model_name="gemini-1.0-pro",
            fallback_model=final_fallback
        )
        primary_model = GeminiChatModel(
            model_name="gemini-pro",
            fallback_model=middle_fallback
        )

        # Mock responses/errors
        mock_primary_error = google_exceptions.InternalServerError("Primary error")
        mock_middle_error = google_exceptions.InternalServerError("Middle error")
        mock_final_response = {
            "content": "Response from final fallback",
            "total_tokens": 5,
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "error": None,
            "model_used": "gpt-3.5-turbo"
        }

        # Setup mocks
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_gemini:
            mock_gemini.side_effect = [mock_primary_error, mock_middle_error]
            
            with patch.object(final_fallback, 'generate_response') as mock_final:
                mock_final.return_value = mock_final_response
                
                # Make the request
                response = primary_model.generate_response(self.test_messages)
                
                # Verify final fallback was used
                self.assertEqual(response["content"], "Response from final fallback")
                self.assertEqual(response["model_used"], "gpt-3.5-turbo")
                self.assertIsNone(response["error"])

    def test_fallback_failure_returns_original_error(self):
        """Test that if fallback fails, we get the original error."""
        # Mock Gemini to fail with a retryable error
        mock_gemini_error = google_exceptions.InternalServerError("Primary error")
        
        # Mock OpenAI to also fail
        mock_openai_error = {
            "content": "",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": ModelError(
                type="server_error",
                message="OpenAI server error",
                retryable=True
            ),
            "model_used": "gpt-4"
        }
        
        # Setup mocks
        with patch('google.generativeai.GenerativeModel.start_chat') as mock_gemini:
            mock_gemini.side_effect = mock_gemini_error
            
            with patch.object(self.fallback_model, 'generate_response') as mock_openai:
                mock_openai.return_value = mock_openai_error
                
                # Make the request
                response = self.primary_model.generate_response(self.test_messages)
                
                # Verify we got the original error
                self.assertEqual(response["content"], "")
                self.assertEqual(response["model_used"], "gemini-pro")
                self.assertEqual(response["error"].type, "server_error")
                self.assertIn("Primary error", response["error"].message)

if __name__ == '__main__':
    unittest.main()