# RecallBricks LangChain Integration - Quickstart Guide

## Installation

```bash
pip install recallbricks-langchain==1.0.0
```

## Prerequisites

1. **RecallBricks API Key**: Get your API key from [recallbricks.com](https://recallbricks.com)
2. **Python 3.8+**: Required for LangChain compatibility
3. **LangChain**: `pip install langchain>=0.1.0`

## Basic Setup

### Environment Variables

Set your API key as an environment variable:

```bash
export RECALLBRICKS_API_KEY="your-api-key-here"
```

Or pass it directly when initializing:

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="my-agent",
    api_key="your-api-key-here"  # Or use RECALLBRICKS_API_KEY env var
)
```

## First Conversation with Memory

### Basic LangChain Integration

```python
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from recallbricks_langchain import RecallBricksMemory

# Initialize memory with RecallBricks
memory = RecallBricksMemory(
    agent_id="my-first-agent",
    api_key="your-recallbricks-api-key",
    organized=True  # Enable organized recall (default)
)

# Create LangChain conversation chain
llm = ChatOpenAI(model="gpt-4")
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Have a conversation - memories are automatically saved
response = conversation.run("Hi! I'm working on a Python project for data analysis.")
print(response)

# Continue the conversation - previous context is recalled
response = conversation.run("What libraries should I use?")
print(response)

# Later, the agent remembers your project context
response = conversation.run("Can you remind me what I'm working on?")
print(response)
```

### Direct Memory Operations

You can also use memory directly without LangChain chains:

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="my-agent",
    api_key="your-api-key"
)

# Save a memory with automatic metadata extraction
result = memory.learn("User prefers dark mode in all applications")
print(f"Saved memory with tags: {result.get('metadata', {}).get('tags', [])}")

# Recall relevant memories
result = memory.recall("What are the user's preferences?")
for mem in result.get("memories", []):
    print(f"- {mem.get('text')}")
```

## Multi-User Applications

For applications with multiple users, use `user_id` to isolate memories:

```python
import uuid

def get_memory_for_user(user_id: str) -> RecallBricksMemory:
    return RecallBricksMemory(
        agent_id="multi-user-agent",
        api_key="your-api-key",
        user_id=user_id  # Must be UUID format
    )

# Each user gets isolated memory
alice_memory = get_memory_for_user(str(uuid.uuid4()))
bob_memory = get_memory_for_user(str(uuid.uuid4()))
```

## Multi-Tenant Applications

For SaaS applications, use `project_id` to isolate by tenant:

```python
memory = RecallBricksMemory(
    agent_id="saas-agent",
    api_key="your-api-key",
    project_id="tenant-abc"  # Isolates memory per tenant
)
```

## Enabling Autonomous Features

For advanced agent capabilities, enable autonomous features:

```python
memory = RecallBricksMemory(
    agent_id="autonomous-agent",
    api_key="your-api-key",
    enable_autonomous=True,
    autonomous_features={
        "working_memory_ttl": 3600,
        "goal_tracking_enabled": True,
        "metacognition_enabled": True,
        "confidence_threshold": 0.7
    }
)

# Now you can use working memory, goal tracking, and metacognition
# See docs/autonomous-features.md for details
```

## Next Steps

- [API Reference](./api-reference.md) - Complete parameter and method documentation
- [Autonomous Features](./autonomous-features.md) - Working memory, goal tracking, metacognition
- [Examples](./examples.md) - Comprehensive code examples

## Troubleshooting

### Common Issues

**API Key Not Found**
```
ValueError: api_key must be provided or RECALLBRICKS_API_KEY environment variable must be set
```
Solution: Set the `RECALLBRICKS_API_KEY` environment variable or pass `api_key` directly.

**Invalid User ID Format**
```
ValueError: user_id must be a valid UUID format
```
Solution: Generate a UUID with `str(uuid.uuid4())`.

**HTTPS Required**
```
ValueError: api_url must use HTTPS for security
```
Solution: Use the default API URL or ensure your custom URL uses HTTPS.

## Support

- Documentation: [recallbricks.com/docs](https://recallbricks.com/docs)
- GitHub: [github.com/recallbricks](https://github.com/recallbricks)
- Email: support@recallbricks.com
