# RecallBricks LangChain Integration - Autonomous Agent Features

Autonomous agent features enable LangChain agents to maintain task-specific context, track progress through goals, and self-assess response quality.

## Enabling Autonomous Features

```python
from recallbricks_langchain import RecallBricksMemory

memory = RecallBricksMemory(
    agent_id="autonomous-agent",
    api_key="your-api-key",
    enable_autonomous=True,
    autonomous_features={
        "working_memory_ttl": 3600,        # 1 hour TTL for sessions
        "goal_tracking_enabled": True,      # Enable goal tracking
        "metacognition_enabled": True,      # Enable quality assessment
        "confidence_threshold": 0.7         # Quality threshold
    }
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `working_memory_ttl` | `int` | `3600` | Time-to-live for working memory sessions (seconds) |
| `goal_tracking_enabled` | `bool` | `True` | Enable goal tracking functionality |
| `metacognition_enabled` | `bool` | `True` | Enable quality and uncertainty assessment |
| `confidence_threshold` | `float` | `0.7` | Minimum confidence for quality assessment |

---

## Working Memory

Working memory provides short-term, task-specific memory that persists only for the duration of a task or session. Ideal for multi-step reasoning.

### create_working_memory_session

Create a new working memory session.

```python
def create_working_memory_session(self, session_id: str) -> Dict[str, Any]
```

**Parameters:**
- `session_id`: Unique identifier for the session

**Returns:**
```python
{
    "session_id": "task-123",
    "created_at": "2024-01-15T10:30:00Z",
    "ttl": 3600,
    "items": [],
    "context": {},
    "active": True
}
```

**Example:**
```python
session = memory.create_working_memory_session("research-task")
print(f"Created session: {session['session_id']}")
```

### add_to_working_memory

Add an item to an existing working memory session.

```python
def add_to_working_memory(
    self,
    session_id: str,
    item: str,
    metadata: Dict[str, Any] = None
) -> None
```

**Parameters:**
- `session_id`: Working memory session ID
- `item`: Content to add
- `metadata`: Optional metadata for the item

**Example:**
```python
memory.add_to_working_memory(
    "research-task",
    "Found 3 relevant documents about async Python",
    metadata={"step": 1, "source": "vector_search"}
)
```

### get_working_memory

Retrieve all items from a working memory session.

```python
def get_working_memory(self, session_id: str) -> List[Dict[str, Any]]
```

**Returns:**
```python
[
    {
        "content": "Found 3 relevant documents",
        "metadata": {"step": 1},
        "added_at": "2024-01-15T10:31:00Z"
    },
    ...
]
```

**Example:**
```python
items = memory.get_working_memory("research-task")
for item in items:
    print(f"Step {item['metadata'].get('step')}: {item['content']}")
```

### end_working_memory_session

End a working memory session with optional persistence.

```python
def end_working_memory_session(
    self,
    session_id: str,
    persist: bool = False
) -> None
```

**Parameters:**
- `session_id`: Working memory session ID
- `persist`: If `True`, save summary to long-term memory

**Example:**
```python
# End without persisting
memory.end_working_memory_session("research-task")

# End and persist to long-term memory
memory.end_working_memory_session("research-task", persist=True)
```

### with_working_memory (Context Manager)

Automatically manage working memory session lifecycle.

```python
@contextmanager
def with_working_memory(self, session_id: str) -> Generator[Dict[str, Any], None, None]
```

**Example:**
```python
with memory.with_working_memory("analysis-task") as session:
    print(f"Session started: {session['session_id']}")

    # Add intermediate results
    memory.add_to_working_memory("analysis-task", "Step 1: Data loaded")
    memory.add_to_working_memory("analysis-task", "Step 2: Analysis complete")

    # Get all results
    items = memory.get_working_memory("analysis-task")

# Session automatically cleaned up here
```

---

## Goal Tracking

Goal tracking enables agents to maintain awareness of their objectives and track progress through multi-step tasks.

### track_goal

Start tracking a new goal with defined steps.

```python
def track_goal(self, goal_id: str, steps: List[str]) -> Dict[str, Any]
```

**Parameters:**
- `goal_id`: Unique identifier for the goal
- `steps`: List of step descriptions

**Returns:**
```python
{
    "goal_id": "search-task",
    "steps": [
        {"step": "Gather requirements", "status": "pending", "completed_at": None},
        {"step": "Search documents", "status": "pending", "completed_at": None},
        {"step": "Synthesize results", "status": "pending", "completed_at": None}
    ],
    "created_at": "2024-01-15T10:30:00Z",
    "current_step": 0,
    "total_steps": 3,
    "status": "in_progress",
    "progress": 0.0
}
```

**Example:**
```python
goal = memory.track_goal("document-search", [
    "Parse user query",
    "Search knowledge base",
    "Rank results",
    "Generate summary"
])
print(f"Tracking goal with {goal['total_steps']} steps")
```

### complete_goal_step

Mark a goal step as complete.

```python
def complete_goal_step(
    self,
    goal_id: str,
    step_index: int = None
) -> Dict[str, Any]
```

**Parameters:**
- `goal_id`: Goal identifier
- `step_index`: Index of step to complete (defaults to current step)

**Returns:** Updated goal data

**Example:**
```python
# Complete current step
status = memory.complete_goal_step("document-search")
print(f"Progress: {status['progress']*100:.0f}%")

# Complete specific step
status = memory.complete_goal_step("document-search", step_index=2)
```

### get_goal_status

Get the current status of a tracked goal.

```python
def get_goal_status(self, goal_id: str) -> Dict[str, Any]
```

**Example:**
```python
status = memory.get_goal_status("document-search")
print(f"Status: {status['status']}")
print(f"Progress: {status['progress']*100:.0f}%")
print(f"Current step: {status['steps'][status['current_step']]['step']}")
```

### with_goal_tracking (Decorator)

Automatically track goal progress for a function.

```python
def with_goal_tracking(self, goal_id: str, steps: List[str])
```

**Example:**
```python
@memory.with_goal_tracking("data-pipeline", ["fetch", "transform", "load"])
def run_pipeline():
    data = fetch_data()
    transformed = transform(data)
    load(transformed)
    return {"status": "complete"}

# Goal tracking starts automatically
result = run_pipeline()
# All steps marked complete on success

# Check final status
status = memory.get_goal_status("data-pipeline")
print(f"Final status: {status['status']}")  # "completed"
```

---

## Metacognition

Metacognition features enable agents to evaluate their own outputs and express appropriate uncertainty.

### assess_quality

Assess the quality of a response.

```python
def assess_quality(
    self,
    response: str,
    confidence: float
) -> Dict[str, Any]
```

**Parameters:**
- `response`: The response text to assess
- `confidence`: Agent's confidence (0.0 to 1.0)

**Returns:**
```python
{
    "response_length": 150,
    "confidence": 0.85,
    "confidence_threshold": 0.7,
    "quality_score": 0.92,
    "meets_quality_bar": True,
    "factors": [
        {"factor": "length", "score": 1.0, "reason": "Response has adequate length"},
        {"factor": "confidence", "score": 1.0, "reason": "Confidence meets threshold"}
    ],
    "recommendations": []
}
```

**Example:**
```python
response = "The capital of France is Paris, located in the north-central part of the country."
assessment = memory.assess_quality(response, confidence=0.95)

if not assessment["meets_quality_bar"]:
    print("Recommendations:")
    for rec in assessment["recommendations"]:
        print(f"  - {rec}")
else:
    print(f"Quality score: {assessment['quality_score']:.2f}")
```

### quantify_uncertainty

Quantify uncertainty in a response.

```python
def quantify_uncertainty(
    self,
    response: str,
    confidence: float,
    evidence: List[str] = None
) -> Dict[str, Any]
```

**Parameters:**
- `response`: The response text to analyze
- `confidence`: Agent's confidence (0.0 to 1.0)
- `evidence`: List of supporting evidence or sources

**Returns:**
```python
{
    "uncertainty_score": 0.25,
    "confidence": 0.75,
    "evidence_count": 2,
    "evidence_strength": 0.75,
    "uncertainty_indicators": ["likely"],
    "knowledge_gaps": [],
    "should_seek_clarification": False,
    "evidence_provided": ["Source 1", "Source 2"]
}
```

**Example:**
```python
response = "The deadline is likely next Friday based on the project timeline."
uncertainty = memory.quantify_uncertainty(
    response=response,
    confidence=0.7,
    evidence=[
        "Email from project manager dated Jan 10",
        "Sprint board showing end date",
        "Team standup notes"
    ]
)

if uncertainty["should_seek_clarification"]:
    print("High uncertainty - consider asking for clarification")
    print(f"Knowledge gaps: {uncertainty['knowledge_gaps']}")
else:
    print(f"Uncertainty score: {uncertainty['uncertainty_score']:.2f}")
    print(f"Evidence strength: {uncertainty['evidence_strength']:.2f}")
```

---

## Status and Monitoring

### get_autonomous_status

Get status of all autonomous agent features.

```python
def get_autonomous_status(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "enabled": True,
    "config": {
        "working_memory_ttl": 3600,
        "goal_tracking_enabled": True,
        "metacognition_enabled": True,
        "confidence_threshold": 0.7
    },
    "working_memory_sessions": 2,
    "active_sessions": ["task-1", "task-2"],
    "active_goals": 1,
    "goals": {
        "search-task": {
            "status": "in_progress",
            "progress": 0.5,
            "current_step": 2
        }
    }
}
```

**Example:**
```python
status = memory.get_autonomous_status()
if status["enabled"]:
    print(f"Active sessions: {status['working_memory_sessions']}")
    print(f"Active goals: {status['active_goals']}")
    for goal_id, info in status["goals"].items():
        print(f"  {goal_id}: {info['progress']*100:.0f}% complete")
```

---

## Complete Example

```python
from recallbricks_langchain import RecallBricksMemory

# Initialize with autonomous features
memory = RecallBricksMemory(
    agent_id="research-agent",
    api_key="your-api-key",
    enable_autonomous=True,
    autonomous_features={
        "working_memory_ttl": 3600,
        "confidence_threshold": 0.7
    }
)

# Perform a multi-step research task
with memory.with_working_memory("research-001") as session:
    # Track the goal
    goal = memory.track_goal("answer-question", [
        "Understand question",
        "Search knowledge base",
        "Synthesize answer",
        "Validate quality"
    ])

    # Step 1: Understand question
    question = "What are the benefits of async Python?"
    memory.add_to_working_memory("research-001", f"Question: {question}")
    memory.complete_goal_step("answer-question")

    # Step 2: Search knowledge base
    results = ["asyncio enables concurrent I/O", "Better performance for I/O-bound tasks"]
    for r in results:
        memory.add_to_working_memory("research-001", f"Found: {r}")
    memory.complete_goal_step("answer-question")

    # Step 3: Synthesize answer
    response = (
        "Async Python provides significant benefits for I/O-bound applications. "
        "The asyncio library enables concurrent execution without threads, "
        "resulting in better performance and resource utilization."
    )
    memory.add_to_working_memory("research-001", f"Response: {response}")
    memory.complete_goal_step("answer-question")

    # Step 4: Validate quality
    assessment = memory.assess_quality(response, confidence=0.85)
    uncertainty = memory.quantify_uncertainty(
        response,
        confidence=0.85,
        evidence=results
    )

    if assessment["meets_quality_bar"]:
        memory.complete_goal_step("answer-question")
        print(f"Response validated with quality score: {assessment['quality_score']:.2f}")
    else:
        print(f"Quality issues: {assessment['recommendations']}")

# Check final status
status = memory.get_goal_status("answer-question")
print(f"Goal status: {status['status']}")  # "completed"
```
