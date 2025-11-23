"""
Basic usage example for RecallBricks LangChain integration.

This example demonstrates:
- Initializing RecallBricksMemory
- Saving conversation context
- Loading memory variables
- How RecallBricks retrieves relevant context

No LLM or ConversationChain required - just the core memory functionality.
"""

from recallbricks_langchain import RecallBricksMemory
import os
import uuid

# Initialize memory with RecallBricks
# Get your service token from the RecallBricks dashboard at https://recallbricks.com
# Note: user_id must be a valid UUID format
memory = RecallBricksMemory(
    agent_id="demo_chatbot",  # Unique identifier for your agent/application
    user_id=str(uuid.uuid4()),  # Generate a UUID for the user
    service_token=os.getenv("RECALLBRICKS_SERVICE_TOKEN")
)

print("=" * 60)
print("RecallBricks Memory - Basic Usage Example")
print("=" * 60)

# Example 1: Save a conversation turn
print("\n1. Saving conversation context...")
memory.save_context(
    {"input": "My name is Alice and I'm working on a chatbot project"},
    {"output": "Nice to meet you, Alice! A chatbot project sounds interesting. What kind of chatbot are you building?"}
)
print("[OK] Saved first conversation turn")

# Example 2: Save another turn
memory.save_context(
    {"input": "I'm using LangChain and RecallBricks for the chatbot"},
    {"output": "Great choice! LangChain provides excellent tools for building LLM applications, and RecallBricks will help your chatbot remember conversations."}
)
print("[OK] Saved second conversation turn")

# Example 3: Load memory for a related query
print("\n2. Loading memory for query: 'What's my name?'")
loaded = memory.load_memory_variables({"input": "What's my name?"})
print(f"\nLoaded memory:\n{loaded['history']}")

# Example 4: Load memory for another query
print("\n" + "=" * 60)
print("3. Loading memory for query: 'What technologies am I using?'")
loaded = memory.load_memory_variables({"input": "What technologies am I using?"})
print(f"\nLoaded memory:\n{loaded['history']}")

# Example 5: Save more context
print("\n" + "=" * 60)
print("4. Saving more conversation context...")
memory.save_context(
    {"input": "The chatbot will help users with customer support"},
    {"output": "That's a valuable use case! Customer support chatbots can greatly improve response times and user satisfaction."}
)
print("[OK] Saved third conversation turn")

# Example 6: Load memory for a complex query
print("\n5. Loading memory for query: 'Tell me about my project'")
loaded = memory.load_memory_variables({"input": "Tell me about my project"})
print(f"\nLoaded memory:\n{loaded['history']}")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
print("\nKey takeaways:")
print("- save_context() stores conversation turns in RecallBricks")
print("- load_memory_variables() retrieves relevant context based on the query")
print("- RecallBricks automatically finds related information across conversations")
print("- No LLM needed to use RecallBricks memory functionality")
print("\nYou can now integrate this with any LangChain chain or use it standalone!")
