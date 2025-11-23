# RecallBricks LangChain Integration

Official LangChain integration for RecallBricks Memory Graph.

## What is RecallBricks?

RecallBricks provides memory infrastructure that goes beyond vector search by understanding **relationships, causality, and patterns** - not just similar text.

Perfect for LangChain agents that need:
- Context across conversations
- Understanding of cause-and-effect
- Relationship-aware memory retrieval

## Installation

```bash
pip install recallbricks-langchain
```

## Quick Start

```python
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from recallbricks_langchain import RecallBricksMemory

# Initialize memory with RecallBricks
memory = RecallBricksMemory(
    api_key="your-recallbricks-api-key",
    user_id="user-123"  # Optional: for multi-user apps
)

# Use with any LangChain chain
llm = ChatOpenAI()
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# Your agent now has relationship-aware memory!
conversation.run("Deploy the new authentication feature")
# Later...
conversation.run("Why did the deployment fail?")
# RecallBricks provides context: "Related to the auth feature you deployed"
```

## Features

✅ **Drop-in replacement** for ConversationBufferMemory
✅ **Automatic relationship detection** - understands causality and patterns
✅ **Persistent across sessions** - memories don't disappear
✅ **Multi-user support** - isolate memory per user
✅ **Production-ready** - 99.9% uptime, enterprise security

## Why RecallBricks vs Standard LangChain Memory?

| Feature | ConversationBufferMemory | RecallBricksMemory |
|---------|--------------------------|-------------------|
| Stores conversations | ✅ | ✅ |
| Persists across sessions | ❌ | ✅ |
| Understands relationships | ❌ | ✅ |
| Detects causality | ❌ | ✅ |
| Finds patterns | ❌ | ✅ |
| Explains connections | ❌ | ✅ |

## Advanced Usage

### With Custom Parameters

```python
memory = RecallBricksMemory(
    api_key="your-key",
    user_id="user-123",
    limit=20,  # Number of memories to retrieve
    min_relevance=0.7,  # Minimum relevance score
    return_messages=True  # Return as Message objects
)
```

### Multi-User Applications

```python
def get_conversation_for_user(user_id: str):
    memory = RecallBricksMemory(
        api_key="your-key",
        user_id=user_id  # Isolates memory per user
    )
    return ConversationChain(llm=llm, memory=memory)
```

## Get Your API Key

1. Sign up at [recallbricks.com](https://recallbricks.com)
2. Get your API key from the dashboard
3. Start building!

## Documentation

- [RecallBricks Docs](https://recallbricks.com/docs)
- [API Reference](https://recallbricks.com/docs#api-reference)
- [LangChain Docs](https://python.langchain.com/docs/modules/memory/)

## Examples

Check out the `examples/` directory for:
- `basic_usage.py` - Simple conversation example
- `with_openai.py` - Advanced multi-user scenarios with relationship detection

## Development

### Installation

```bash
# Clone the repository
git clone https://github.com/recallbricks/recallbricks-langchain.git
cd recallbricks-langchain

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
python -m pytest tests/
```

## Support

- Email: support@recallbricks.com
- GitHub: [github.com/recallbricks](https://github.com/recallbricks)

## License

MIT
