# RecallBricks LangChain Integration - Examples

## Basic Conversation Chain with Memory

```python
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from recallbricks_langchain import RecallBricksMemory
import os

# Initialize memory
memory = RecallBricksMemory(
    agent_id="conversation-agent",
    api_key=os.getenv("RECALLBRICKS_API_KEY"),
    organized=True  # Enable organized recall
)

# Create conversation chain
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Have a conversation
response = conversation.run("Hi! I'm building a data pipeline in Python.")
print(f"AI: {response}")

response = conversation.run("What tools would you recommend?")
print(f"AI: {response}")

# Later - the agent remembers context
response = conversation.run("Remind me what I'm working on.")
print(f"AI: {response}")
```

---

## Using Autonomous Features

### Working Memory Session

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="autonomous-agent",
    api_key="your-api-key",
    enable_autonomous=True,
    autonomous_features={
        "working_memory_ttl": 3600,
        "goal_tracking_enabled": True,
        "metacognition_enabled": True
    }
)

# Use context manager for automatic cleanup
with memory.with_working_memory("data-analysis-task") as session:
    print(f"Started session: {session['session_id']}")

    # Step 1: Load data
    memory.add_to_working_memory(
        "data-analysis-task",
        "Loaded 10,000 rows from sales database",
        metadata={"step": "load", "rows": 10000}
    )

    # Step 2: Transform data
    memory.add_to_working_memory(
        "data-analysis-task",
        "Aggregated by region: 5 regions found",
        metadata={"step": "transform", "regions": 5}
    )

    # Step 3: Analyze
    memory.add_to_working_memory(
        "data-analysis-task",
        "Top region: West Coast with $2.5M revenue",
        metadata={"step": "analyze", "top_region": "West Coast"}
    )

    # Review all working memory
    items = memory.get_working_memory("data-analysis-task")
    print("\nWorking Memory Contents:")
    for item in items:
        print(f"  [{item['metadata'].get('step')}] {item['content']}")

# Session automatically cleaned up
print("\nSession ended and cleaned up")
```

---

### Goal Tracking Example

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="task-agent",
    api_key="your-api-key",
    enable_autonomous=True
)

# Define a multi-step goal
goal = memory.track_goal("customer-support-ticket", [
    "Parse customer inquiry",
    "Search knowledge base",
    "Draft response",
    "Review for quality",
    "Send to customer"
])

print(f"Started goal: {goal['goal_id']}")
print(f"Total steps: {goal['total_steps']}")

# Simulate completing steps
steps_completed = [
    "Parsed: Customer asking about refund policy",
    "Found 3 relevant KB articles",
    "Drafted response with refund process",
    "Quality check passed",
    "Response sent successfully"
]

for i, step_result in enumerate(steps_completed):
    print(f"\nStep {i+1}: {step_result}")
    status = memory.complete_goal_step("customer-support-ticket")
    print(f"  Progress: {status['progress']*100:.0f}%")

# Final status
final = memory.get_goal_status("customer-support-ticket")
print(f"\nGoal completed: {final['status']}")
print(f"Completed at: {final.get('completed_at')}")
```

---

### Goal Tracking with Decorator

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="pipeline-agent",
    api_key="your-api-key",
    enable_autonomous=True
)

@memory.with_goal_tracking("etl-pipeline", [
    "Extract from source",
    "Transform data",
    "Load to destination",
    "Validate results"
])
def run_etl_pipeline(source: str, destination: str):
    """ETL pipeline with automatic goal tracking."""
    print(f"Running pipeline: {source} -> {destination}")

    # Simulate pipeline work
    extracted = {"records": 1000}
    transformed = {"records": 1000, "columns": 15}
    loaded = {"records": 1000}

    return {
        "source": source,
        "destination": destination,
        "records_processed": 1000,
        "status": "success"
    }

# Run the pipeline - goal tracking happens automatically
result = run_etl_pipeline("sales_db", "warehouse")
print(f"\nPipeline result: {result}")

# Check goal status
status = memory.get_goal_status("etl-pipeline")
print(f"Goal status: {status['status']}")  # "completed"
```

---

### Metacognition: Quality Assessment

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="qa-agent",
    api_key="your-api-key",
    enable_autonomous=True,
    autonomous_features={"confidence_threshold": 0.7}
)

# Test different response qualities
responses = [
    {
        "text": "Yes.",
        "confidence": 0.5,
        "description": "Too brief, low confidence"
    },
    {
        "text": "The capital of France is Paris, a city known for the Eiffel Tower.",
        "confidence": 0.95,
        "description": "Good length, high confidence"
    },
    {
        "text": "I think maybe it could be around 5, possibly.",
        "confidence": 0.3,
        "description": "Uncertain language, low confidence"
    }
]

for resp in responses:
    print(f"\n{'='*50}")
    print(f"Response: \"{resp['text']}\"")
    print(f"Expected: {resp['description']}")

    assessment = memory.assess_quality(resp['text'], resp['confidence'])

    print(f"\nAssessment:")
    print(f"  Quality Score: {assessment['quality_score']:.2f}")
    print(f"  Meets Bar: {assessment['meets_quality_bar']}")
    print(f"  Factors:")
    for factor in assessment['factors']:
        print(f"    - {factor['factor']}: {factor['score']:.2f} ({factor['reason']})")
    if assessment['recommendations']:
        print(f"  Recommendations:")
        for rec in assessment['recommendations']:
            print(f"    - {rec}")
```

---

### Metacognition: Uncertainty Quantification

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="research-agent",
    api_key="your-api-key",
    enable_autonomous=True
)

# Response with strong evidence
strong_response = memory.quantify_uncertainty(
    response="The product launch is scheduled for March 15th.",
    confidence=0.9,
    evidence=[
        "Official press release dated Feb 1st",
        "CEO announcement at company meeting",
        "Marketing calendar confirmation",
        "Pre-order page live with March 15th date"
    ]
)

print("Strong Evidence Response:")
print(f"  Uncertainty Score: {strong_response['uncertainty_score']:.2f}")
print(f"  Evidence Strength: {strong_response['evidence_strength']:.2f}")
print(f"  Should Clarify: {strong_response['should_seek_clarification']}")

# Response with weak evidence
weak_response = memory.quantify_uncertainty(
    response="The project might possibly be delayed, I'm not entirely sure.",
    confidence=0.3,
    evidence=[]
)

print("\nWeak Evidence Response:")
print(f"  Uncertainty Score: {weak_response['uncertainty_score']:.2f}")
print(f"  Evidence Strength: {weak_response['evidence_strength']:.2f}")
print(f"  Should Clarify: {weak_response['should_seek_clarification']}")
print(f"  Uncertainty Indicators: {weak_response['uncertainty_indicators']}")
print(f"  Knowledge Gaps: {weak_response['knowledge_gaps']}")
```

---

## Multi-User Application

```python
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from recallbricks_langchain import RecallBricksMemory
import uuid

def create_user_conversation(user_id: str, user_name: str):
    """Create an isolated conversation for a specific user."""
    memory = RecallBricksMemory(
        agent_id="multi-user-support",
        api_key="your-api-key",
        user_id=user_id  # Must be UUID format
    )

    llm = ChatOpenAI(model="gpt-4")
    return ConversationChain(llm=llm, memory=memory)

# Simulate two users
alice_id = str(uuid.uuid4())
bob_id = str(uuid.uuid4())

alice_conv = create_user_conversation(alice_id, "Alice")
bob_conv = create_user_conversation(bob_id, "Bob")

# Alice's conversation
alice_conv.run("Hi, I'm Alice. I need help with Python async programming.")
alice_conv.run("Specifically, I'm building a web scraper.")

# Bob's conversation (completely isolated)
bob_conv.run("Hello, I'm Bob. I'm working on a machine learning model.")
bob_conv.run("I'm using PyTorch for image classification.")

# Each user's context is isolated
alice_response = alice_conv.run("What am I working on?")
print(f"Alice's context: {alice_response}")

bob_response = bob_conv.run("What am I working on?")
print(f"Bob's context: {bob_response}")
```

---

## RAG with RecallBricksRetriever

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from recallbricks_langchain import RecallBricksRetriever

# Initialize retriever
retriever = RecallBricksRetriever(
    api_key="your-api-key",
    k=5,  # Return top 5 results
    organized=True  # Include category summaries
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query the knowledge base
result = qa_chain({"query": "What is the company refund policy?"})
print(f"Answer: {result['result']}")
print(f"Sources: {len(result['source_documents'])} documents")
```

---

## Monitoring and Health Checks

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="monitored-agent",
    api_key="your-api-key",
    enable_metrics=True
)

# Perform some operations
memory.learn("Test memory 1")
memory.learn("Test memory 2")
memory.recall("test")

# Get metrics
metrics = memory.get_detailed_metrics()
print("Metrics:")
print(f"  Total Requests: {metrics.get('requests_total', 0)}")
print(f"  Success Rate: {metrics.get('success_rate', 0)*100:.1f}%")
print(f"  Avg Response Time: {metrics.get('response_time_avg', 0)*1000:.0f}ms")
print(f"  P95 Response Time: {metrics.get('response_time_p95', 0)*1000:.0f}ms")

# Health check
health = memory.health_check()
print(f"\nHealth Status: {health['status']}")
for check_name, check_data in health['checks'].items():
    print(f"  {check_name}: {check_data['status']}")

# Circuit breaker status
cb_status = memory.get_circuit_breaker_status()
print(f"\nCircuit Breaker: {cb_status['state']}")

# Prometheus metrics export
prometheus = memory.get_prometheus_metrics()
print(f"\nPrometheus metrics ({len(prometheus)} chars)")
```

---

## Complete Autonomous Agent Example

```python
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from recallbricks_langchain import RecallBricksMemory
import os

def autonomous_research_agent(query: str):
    """Complete autonomous agent with all features."""

    # Initialize with all autonomous features
    memory = RecallBricksMemory(
        agent_id="research-agent",
        api_key=os.getenv("RECALLBRICKS_API_KEY"),
        enable_autonomous=True,
        autonomous_features={
            "working_memory_ttl": 3600,
            "goal_tracking_enabled": True,
            "metacognition_enabled": True,
            "confidence_threshold": 0.7
        },
        organized=True,
        enable_metrics=True
    )

    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    conversation = ConversationChain(llm=llm, memory=memory)

    # Start working memory for this research task
    with memory.with_working_memory(f"research-{hash(query)}") as session:

        # Track the research goal
        goal = memory.track_goal("research-query", [
            "Analyze query",
            "Search knowledge base",
            "Generate response",
            "Validate quality"
        ])

        # Step 1: Analyze query
        memory.add_to_working_memory(session["session_id"], f"Query: {query}")
        memory.complete_goal_step("research-query")

        # Step 2: Search knowledge base (handled by memory.recall internally)
        context = memory.recall(query, limit=10)
        memory.add_to_working_memory(
            session["session_id"],
            f"Found {len(context.get('memories', []))} relevant memories"
        )
        memory.complete_goal_step("research-query")

        # Step 3: Generate response
        response = conversation.run(query)
        memory.add_to_working_memory(session["session_id"], f"Response: {response[:100]}...")
        memory.complete_goal_step("research-query")

        # Step 4: Validate quality
        assessment = memory.assess_quality(response, confidence=0.8)
        uncertainty = memory.quantify_uncertainty(
            response,
            confidence=0.8,
            evidence=[m.get("text", "")[:50] for m in context.get("memories", [])[:3]]
        )

        if assessment["meets_quality_bar"] and not uncertainty["should_seek_clarification"]:
            memory.complete_goal_step("research-query")
            quality_status = "passed"
        else:
            quality_status = "needs_review"

        # Get final status
        goal_status = memory.get_goal_status("research-query")

        return {
            "response": response,
            "quality_score": assessment["quality_score"],
            "uncertainty_score": uncertainty["uncertainty_score"],
            "quality_status": quality_status,
            "goal_progress": goal_status["progress"],
            "goal_status": goal_status["status"]
        }

# Run the agent
if __name__ == "__main__":
    result = autonomous_research_agent("What are best practices for async Python?")
    print(f"Response: {result['response'][:200]}...")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"Uncertainty: {result['uncertainty_score']:.2f}")
    print(f"Status: {result['quality_status']}")
    print(f"Goal: {result['goal_status']} ({result['goal_progress']*100:.0f}%)")
```
