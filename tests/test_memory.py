import unittest
from unittest.mock import Mock, patch, MagicMock
from recallbricks_langchain import RecallBricksMemory
from langchain.schema import HumanMessage, AIMessage


class TestRecallBricksMemory(unittest.TestCase):
    """Test suite for RecallBricksMemory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.user_id = "test-user"

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_initialization(self, mock_rb_class):
        """Test that RecallBricksMemory initializes correctly."""
        memory = RecallBricksMemory(
            api_key=self.api_key,
            user_id=self.user_id,
            limit=20,
            min_relevance=0.7
        )

        # Verify RecallBricks was initialized with correct API key
        mock_rb_class.assert_called_once_with(self.api_key)

        # Verify attributes are set correctly
        self.assertEqual(memory.user_id, self.user_id)
        self.assertEqual(memory.limit, 20)
        self.assertEqual(memory.min_relevance, 0.7)
        self.assertFalse(memory.return_messages)

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_memory_variables(self, mock_rb_class):
        """Test that memory_variables returns correct list."""
        memory = RecallBricksMemory(api_key=self.api_key)
        self.assertEqual(memory.memory_variables, ["history"])

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_save_context(self, mock_rb_class):
        """Test that save_context correctly saves to RecallBricks."""
        mock_rb = Mock()
        mock_rb_class.return_value = mock_rb

        memory = RecallBricksMemory(
            api_key=self.api_key,
            user_id=self.user_id
        )

        inputs = {"input": "What is my name?"}
        outputs = {"output": "Your name is Alice."}

        memory.save_context(inputs, outputs)

        # Verify create_memory was called with correct arguments
        mock_rb.create_memory.assert_called_once_with(
            text="Human: What is my name?\nAI: Your name is Alice.",
            tags=["conversation", self.user_id],
            metadata={
                "user_id": self.user_id,
                "type": "conversation_turn"
            }
        )

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_save_context_without_user_id(self, mock_rb_class):
        """Test save_context when no user_id is provided."""
        mock_rb = Mock()
        mock_rb_class.return_value = mock_rb

        memory = RecallBricksMemory(api_key=self.api_key)

        inputs = {"input": "Hello"}
        outputs = {"output": "Hi there!"}

        memory.save_context(inputs, outputs)

        # Verify tags don't include user_id when it's None
        call_args = mock_rb.create_memory.call_args
        self.assertEqual(call_args[1]["tags"], ["conversation"])
        self.assertIsNone(call_args[1]["metadata"]["user_id"])

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_load_memory_variables_as_string(self, mock_rb_class):
        """Test loading memory variables as string format."""
        mock_rb = Mock()
        mock_rb_class.return_value = mock_rb

        # Mock search results
        mock_result1 = Mock()
        mock_result1.text = "Human: My name is Alice\nAI: Nice to meet you, Alice!"
        mock_result2 = Mock()
        mock_result2.text = "Human: I like Python\nAI: Python is a great language!"

        mock_rb.search.return_value = [mock_result1, mock_result2]

        memory = RecallBricksMemory(
            api_key=self.api_key,
            user_id=self.user_id,
            return_messages=False
        )

        inputs = {"input": "What do you know about me?"}
        result = memory.load_memory_variables(inputs)

        # Verify search was called correctly
        mock_rb.search.assert_called_once_with(
            query="What do you know about me?",
            limit=10,
            tags=[self.user_id],
            include_relationships=True
        )

        # Verify result format
        expected_history = (
            "Human: My name is Alice\nAI: Nice to meet you, Alice!\n\n"
            "Human: I like Python\nAI: Python is a great language!"
        )
        self.assertEqual(result["history"], expected_history)

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_load_memory_variables_as_messages(self, mock_rb_class):
        """Test loading memory variables as message objects."""
        mock_rb = Mock()
        mock_rb_class.return_value = mock_rb

        # Mock search results
        mock_result = Mock()
        mock_result.text = "Human: My name is Alice\nAI: Nice to meet you, Alice!"

        mock_rb.search.return_value = [mock_result]

        memory = RecallBricksMemory(
            api_key=self.api_key,
            user_id=self.user_id,
            return_messages=True
        )

        inputs = {"input": "What is my name?"}
        result = memory.load_memory_variables(inputs)

        # Verify result contains Message objects
        self.assertIn("history", result)
        messages = result["history"]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[0].content, "My name is Alice")
        self.assertEqual(messages[1].content, "Nice to meet you, Alice!")

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_load_memory_variables_without_user_id(self, mock_rb_class):
        """Test loading memory when no user_id is set."""
        mock_rb = Mock()
        mock_rb_class.return_value = mock_rb
        mock_rb.search.return_value = []

        memory = RecallBricksMemory(api_key=self.api_key)

        inputs = {"input": "test query"}
        memory.load_memory_variables(inputs)

        # Verify search was called with tags=None when no user_id
        mock_rb.search.assert_called_once_with(
            query="test query",
            limit=10,
            tags=None,
            include_relationships=True
        )

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_custom_input_output_keys(self, mock_rb_class):
        """Test using custom input/output keys."""
        mock_rb = Mock()
        mock_rb_class.return_value = mock_rb

        memory = RecallBricksMemory(
            api_key=self.api_key,
            input_key="question",
            output_key="answer"
        )

        inputs = {"question": "What is 2+2?"}
        outputs = {"answer": "4"}

        memory.save_context(inputs, outputs)

        # Verify correct keys were used
        call_args = mock_rb.create_memory.call_args
        self.assertEqual(call_args[1]["text"], "Human: What is 2+2?\nAI: 4")

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_clear_method(self, mock_rb_class):
        """Test that clear method doesn't raise errors."""
        memory = RecallBricksMemory(api_key=self.api_key)
        # Should not raise any exceptions
        memory.clear()

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_empty_search_results(self, mock_rb_class):
        """Test handling of empty search results."""
        mock_rb = Mock()
        mock_rb_class.return_value = mock_rb
        mock_rb.search.return_value = []

        memory = RecallBricksMemory(api_key=self.api_key)

        inputs = {"input": "test"}
        result = memory.load_memory_variables(inputs)

        # Should return empty history
        self.assertEqual(result["history"], "")

    @patch('recallbricks_langchain.memory.RecallBricks')
    def test_malformed_conversation_text(self, mock_rb_class):
        """Test handling of malformed conversation text in messages mode."""
        mock_rb = Mock()
        mock_rb_class.return_value = mock_rb

        # Mock result without proper format
        mock_result = Mock()
        mock_result.text = "Just some text without markers"

        mock_rb.search.return_value = [mock_result]

        memory = RecallBricksMemory(
            api_key=self.api_key,
            return_messages=True
        )

        inputs = {"input": "test"}
        result = memory.load_memory_variables(inputs)

        # Should handle gracefully and return empty messages list
        self.assertEqual(result["history"], [])


if __name__ == '__main__':
    unittest.main()
