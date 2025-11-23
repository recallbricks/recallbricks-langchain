"""
Advanced example: Multi-user application with OpenAI.

This example demonstrates:
- Using RecallBricksMemory with OpenAI in a multi-user scenario
- Isolating memory per user
- Custom configuration options
- Relationship-aware context retrieval
"""

from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from recallbricks_langchain import RecallBricksMemory
import os


def get_conversation_for_user(user_id: str) -> ConversationChain:
    """
    Create a conversation chain with isolated memory for a specific user.

    Args:
        user_id: Unique identifier for the user

    Returns:
        ConversationChain configured with RecallBricks memory
    """
    # Get your service token from the RecallBricks dashboard at https://recallbricks.com
    memory = RecallBricksMemory(
        agent_id="multi_user_app",  # Unique identifier for your agent/application
        user_id=user_id,
        service_token=os.getenv("RECALLBRICKS_SERVICE_TOKEN"),
        api_url="https://recallbricks-api-clean.onrender.com",  # Optional: defaults to production API
        limit=20,  # Retrieve up to 20 relevant memories
        min_relevance=0.7,  # Only return highly relevant memories
        return_messages=True  # Return as Message objects for better control
    )

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7
    )

    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )


def simulate_user_session(user_id: str, name: str):
    """Simulate a conversation session for a user.

    Note: user_id must be a valid UUID format.
    """
    print(f"\n{'=' * 60}")
    print(f"Session for User: {name} (ID: {user_id})")
    print(f"{'=' * 60}\n")

    conversation = get_conversation_for_user(user_id)

    # First interaction
    print(f"[{name}] Introducing themselves...")
    response = conversation.run(
        f"Hi! I'm {name}. I'm building a web application with React."
    )
    print(f"AI: {response}\n")

    # Second interaction
    print(f"[{name}] Asking about database...")
    response = conversation.run(
        "I'm trying to decide between PostgreSQL and MongoDB for my database."
    )
    print(f"AI: {response}\n")

    # Third interaction - testing memory recall
    print(f"[{name}] Testing memory...")
    response = conversation.run(
        "What do you know about my project?"
    )
    print(f"AI: {response}\n")


def demonstrate_multi_user():
    """Demonstrate multi-user memory isolation."""
    import uuid

    print("\n" + "=" * 60)
    print("MULTI-USER DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows how RecallBricks isolates memory per user.")
    print("Each user gets their own context, even in the same application.\n")

    # Simulate two different users with valid UUIDs
    alice_id = str(uuid.uuid4())
    bob_id = str(uuid.uuid4())

    simulate_user_session(alice_id, "Alice")
    simulate_user_session(bob_id, "Bob")

    # Now demonstrate that memories are isolated
    print("\n" + "=" * 60)
    print("TESTING MEMORY ISOLATION")
    print("=" * 60)

    # Alice's conversation should only know about Alice
    print("\nAsking Alice's conversation about Bob...")
    alice_conversation = get_conversation_for_user(alice_id)
    response = alice_conversation.run("What do you know about Bob?")
    print(f"AI: {response}\n")

    # Bob's conversation should only know about Bob
    print("Asking Bob's conversation about Alice...")
    bob_conversation = get_conversation_for_user(bob_id)
    response = bob_conversation.run("What do you know about Alice?")
    print(f"AI: {response}\n")

    print("=" * 60)
    print("Notice how each user's memory is completely isolated!")
    print("=" * 60)


def demonstrate_relationship_awareness():
    """Demonstrate RecallBricks' relationship detection."""
    import uuid

    print("\n" + "=" * 60)
    print("RELATIONSHIP AWARENESS DEMONSTRATION")
    print("=" * 60)
    print("\nRecallBricks automatically detects relationships between memories.")
    print("Watch how it connects cause-and-effect:\n")

    # Use a valid UUID for the demo user
    demo_user_id = str(uuid.uuid4())
    conversation = get_conversation_for_user(demo_user_id)

    # Create a sequence of related events
    print("1. Reporting an issue...")
    conversation.run("I deployed the authentication feature to production")

    print("\n2. Noting a problem...")
    conversation.run("Users are reporting that they can't log in")

    print("\n3. Finding the cause...")
    conversation.run("I found a bug in the JWT token validation")

    print("\n4. Fixing it...")
    conversation.run("I fixed the JWT bug and redeployed")

    # Now query about the problem - RecallBricks should understand the full chain
    print("\n5. Asking about the issue...")
    response = conversation.run("Why were users unable to log in?")
    print(f"AI: {response}\n")

    print("=" * 60)
    print("RecallBricks connected:")
    print("- The authentication deployment")
    print("- The login issue")
    print("- The root cause (JWT bug)")
    print("- The fix")
    print("This is relationship-aware memory in action!")
    print("=" * 60)


if __name__ == "__main__":
    # Check for required API keys
    if not os.getenv("RECALLBRICKS_SERVICE_TOKEN"):
        print("Error: RECALLBRICKS_SERVICE_TOKEN environment variable not set")
        print("Get your service token from the RecallBricks dashboard at: https://recallbricks.com")
        exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    # Run demonstrations
    try:
        demonstrate_multi_user()
        demonstrate_relationship_awareness()

        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("✓ RecallBricks provides persistent, relationship-aware memory")
        print("✓ Memory is automatically isolated per user")
        print("✓ Relationships and causality are detected automatically")
        print("✓ Drop-in replacement for standard LangChain memory")
        print("\nLearn more at: https://recallbricks.com/docs")

    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Make sure you have valid API keys set.")
