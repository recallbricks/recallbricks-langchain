"""
Unit tests for RecallBricksMemory class (v2.0).

Tests the core memory functionality including learn, recall,
save_context, and load_memory_variables.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import uuid

from recallbricks_langchain import RecallBricksMemory
from langchain_core.messages import HumanMessage, AIMessage


class MockResponse:
    """Mock HTTP response object."""

    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(json_data) if json_data else ""
        self.content = self.text.encode() if self.text else b""

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if not self.ok:
            from requests import HTTPError
            raise HTTPError(f"HTTP Error: {self.status_code}")


class TestRecallBricksMemory(unittest.TestCase):
    """Test suite for RecallBricksMemory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.agent_id = "test-agent"
        self.user_id = str(uuid.uuid4())  # Valid UUID format

    @patch('recallbricks_langchain.memory.get_session')
    def test_initialization(self, mock_get_session):
        """Test that RecallBricksMemory initializes correctly."""
        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            user_id=self.user_id,
            api_key=self.api_key,
            limit=20,
            min_relevance=0.7,
            enable_logging=False
        )

        # Verify attributes are set correctly
        self.assertEqual(memory.agent_id, self.agent_id)
        self.assertEqual(memory.user_id, self.user_id)
        self.assertEqual(memory.limit, 20)
        self.assertEqual(memory.min_relevance, 0.7)
        self.assertFalse(memory.return_messages)

    @patch('recallbricks_langchain.memory.get_session')
    def test_memory_variables(self, mock_get_session):
        """Test that memory_variables returns correct list."""
        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            api_key=self.api_key,
            enable_logging=False
        )
        self.assertEqual(memory.memory_variables, ["history"])

    @patch('recallbricks_langchain.memory.get_session')
    def test_save_context(self, mock_get_session):
        """Test that save_context correctly saves to RecallBricks."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.post.return_value = MockResponse({"id": "mem-123"})

        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            user_id=self.user_id,
            api_key=self.api_key,
            enable_logging=False
        )

        inputs = {"input": "What is my name?"}
        outputs = {"output": "Your name is Alice."}

        memory.save_context(inputs, outputs)

        # Verify POST was called (twice for input and output)
        self.assertEqual(mock_session.post.call_count, 2)

        # Check the first call (input)
        first_call = mock_session.post.call_args_list[0]
        self.assertIn("User: What is my name?", first_call[1]["json"]["text"])

        # Check the second call (output)
        second_call = mock_session.post.call_args_list[1]
        self.assertIn("Assistant: Your name is Alice.", second_call[1]["json"]["text"])

    @patch('recallbricks_langchain.memory.get_session')
    def test_save_context_without_user_id(self, mock_get_session):
        """Test save_context when no user_id is provided."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.post.return_value = MockResponse({"id": "mem-123"})

        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            api_key=self.api_key,
            enable_logging=False
        )

        inputs = {"input": "Hello"}
        outputs = {"output": "Hi there!"}

        memory.save_context(inputs, outputs)

        # Should still work without user_id
        self.assertEqual(mock_session.post.call_count, 2)

    @patch('recallbricks_langchain.memory.get_session')
    def test_load_memory_variables_as_string(self, mock_get_session):
        """Test loading memory variables as string format."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        # Mock recall response
        mock_session.post.return_value = MockResponse({
            "memories": [
                {
                    "id": "mem-1",
                    "text": "My name is Alice",
                    "score": 0.95,
                    "metadata": {"category": "Personal Info"}
                },
                {
                    "id": "mem-2",
                    "text": "I like Python",
                    "score": 0.90,
                    "metadata": {"category": "Preferences"}
                }
            ],
            "categories": {}
        })

        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            user_id=self.user_id,
            api_key=self.api_key,
            return_messages=False,
            organized=False,
            enable_logging=False
        )

        inputs = {"input": "What do you know about me?"}
        result = memory.load_memory_variables(inputs)

        # Verify recall was called
        mock_session.post.assert_called()

        # Verify result format (list with context string)
        self.assertIn("history", result)
        history = result["history"]
        self.assertIsInstance(history, list)
        if history:
            self.assertIn("My name is Alice", history[0])

    @patch('recallbricks_langchain.memory.get_session')
    def test_load_memory_variables_as_messages(self, mock_get_session):
        """Test loading memory variables as message objects."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        # Mock recall response
        mock_session.post.return_value = MockResponse({
            "memories": [
                {
                    "id": "mem-1",
                    "text": "My name is Alice",
                    "score": 0.95,
                    "metadata": {}
                }
            ],
            "categories": {}
        })

        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            user_id=self.user_id,
            api_key=self.api_key,
            return_messages=True,
            enable_logging=False
        )

        inputs = {"input": "What is my name?"}
        result = memory.load_memory_variables(inputs)

        # Verify result contains Message objects
        self.assertIn("history", result)
        messages = result["history"]
        self.assertIsInstance(messages, list)
        if messages:
            self.assertIsInstance(messages[0], HumanMessage)

    @patch('recallbricks_langchain.memory.get_session')
    def test_load_memory_variables_without_user_id(self, mock_get_session):
        """Test loading memory when no user_id is set."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.post.return_value = MockResponse({
            "memories": [],
            "categories": {}
        })

        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            api_key=self.api_key,
            enable_logging=False
        )

        inputs = {"input": "test query"}
        result = memory.load_memory_variables(inputs)

        # Should work without user_id
        self.assertIn("history", result)

    @patch('recallbricks_langchain.memory.get_session')
    def test_custom_input_output_keys(self, mock_get_session):
        """Test using custom input/output keys."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.post.return_value = MockResponse({"id": "mem-123"})

        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            api_key=self.api_key,
            input_key="question",
            output_key="answer",
            enable_logging=False
        )

        inputs = {"question": "What is 2+2?"}
        outputs = {"answer": "4"}

        memory.save_context(inputs, outputs)

        # Verify correct keys were used
        calls = mock_session.post.call_args_list
        self.assertIn("User: What is 2+2?", calls[0][1]["json"]["text"])
        self.assertIn("Assistant: 4", calls[1][1]["json"]["text"])

    @patch('recallbricks_langchain.memory.get_session')
    def test_clear_method(self, mock_get_session):
        """Test that clear method doesn't raise errors."""
        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            api_key=self.api_key,
            enable_logging=False
        )
        # Should not raise any exceptions
        memory.clear()

    @patch('recallbricks_langchain.memory.get_session')
    def test_empty_search_results(self, mock_get_session):
        """Test handling of empty search results."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.post.return_value = MockResponse({
            "memories": [],
            "categories": {}
        })

        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            api_key=self.api_key,
            enable_logging=False
        )

        inputs = {"input": "test"}
        result = memory.load_memory_variables(inputs)

        # Should return empty history list
        self.assertEqual(result["history"], [])

    @patch('recallbricks_langchain.memory.get_session')
    def test_malformed_conversation_text(self, mock_get_session):
        """Test handling of malformed conversation text in messages mode."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session

        # Mock result without proper format
        mock_session.post.return_value = MockResponse({
            "memories": [
                {
                    "id": "mem-1",
                    "text": "Just some text without markers",
                    "score": 0.9,
                    "metadata": {}
                }
            ],
            "categories": {}
        })

        memory = RecallBricksMemory(
            agent_id=self.agent_id,
            api_key=self.api_key,
            return_messages=True,
            enable_logging=False
        )

        inputs = {"input": "test"}
        result = memory.load_memory_variables(inputs)

        # Should handle gracefully - text is still converted to HumanMessage
        self.assertIn("history", result)
        messages = result["history"]
        self.assertIsInstance(messages, list)
        # The text will be wrapped as HumanMessage
        if messages:
            self.assertIsInstance(messages[0], HumanMessage)

    def test_validation_agent_id_required(self):
        """Test that agent_id is required."""
        with self.assertRaises(ValueError):
            RecallBricksMemory(
                agent_id="",
                api_key=self.api_key
            )

    def test_validation_api_key_required(self):
        """Test that api_key is required when env var not set."""
        import os
        original = os.environ.pop("RECALLBRICKS_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                RecallBricksMemory(agent_id=self.agent_id)
        finally:
            if original:
                os.environ["RECALLBRICKS_API_KEY"] = original

    def test_validation_https_required(self):
        """Test that HTTPS is required for API URL."""
        with self.assertRaises(ValueError):
            RecallBricksMemory(
                agent_id=self.agent_id,
                api_key=self.api_key,
                api_url="http://insecure.com"
            )

    def test_validation_user_id_uuid_format(self):
        """Test that user_id must be valid UUID format."""
        with self.assertRaises(ValueError):
            RecallBricksMemory(
                agent_id=self.agent_id,
                user_id="not-a-valid-uuid",
                api_key=self.api_key
            )


if __name__ == '__main__':
    unittest.main()
