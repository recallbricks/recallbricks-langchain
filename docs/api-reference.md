# RecallBricks LangChain Integration - API Reference

## RecallBricksMemory

Enterprise-grade memory class for LangChain using RecallBricks.

```python
from recallbricks_langchain import RecallBricksMemory
```

### Constructor

```python
RecallBricksMemory(
    agent_id: str,
    user_id: str = None,
    api_key: str = None,
    api_url: str = "https://api.recallbricks.com/api/v1",
    return_messages: bool = False,
    input_key: str = None,
    output_key: str = None,
    limit: int = 10,
    min_relevance: float = 0.6,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_timeout: int = 60,
    enable_logging: bool = True,
    max_text_length: int = 100000,
    rate_limit: int = 100,
    rate_limit_period: int = 60,
    enable_deduplication: bool = True,
    enable_metrics: bool = True,
    enable_distributed_tracing: bool = True,
    organized: bool = True,
    source: str = "langchain",
    project_id: str = None,
    enable_autonomous: bool = False,
    autonomous_features: Dict[str, Any] = None
)
```

### Parameters

#### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_id` | `str` | Unique identifier for your agent/application. Required. |

#### Optional Parameters - Core

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | `str` | `None` | User ID for multi-user applications. Must be UUID format. |
| `api_key` | `str` | `None` | RecallBricks API key. Falls back to `RECALLBRICKS_API_KEY` env var. |
| `api_url` | `str` | `"https://api.recallbricks.com/api/v1"` | API base URL. Must use HTTPS. |
| `project_id` | `str` | `None` | Project ID for multi-tenant applications. |

#### Optional Parameters - LangChain Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_messages` | `bool` | `False` | Return as Message objects instead of strings. |
| `input_key` | `str` | `None` | Key to extract input from. Defaults to `"input"`. |
| `output_key` | `str` | `None` | Key to extract output from. Defaults to `"output"`. |
| `limit` | `int` | `10` | Maximum number of memories to retrieve (1-1000). |
| `min_relevance` | `float` | `0.6` | Minimum relevance score (0.0-1.0). |
| `organized` | `bool` | `True` | Use organized recall with category summaries. |
| `source` | `str` | `"langchain"` | Source identifier for saved memories. |

#### Optional Parameters - Reliability

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `int` | `3` | Maximum retry attempts for failed requests. |
| `retry_delay` | `float` | `1.0` | Base delay for exponential backoff (seconds). |
| `circuit_breaker_threshold` | `int` | `5` | Failures before circuit breaker opens. |
| `circuit_breaker_timeout` | `int` | `60` | Seconds before attempting recovery. |
| `rate_limit` | `int` | `100` | Maximum requests per period. |
| `rate_limit_period` | `int` | `60` | Rate limit period in seconds. |

#### Optional Parameters - Features

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_logging` | `bool` | `True` | Enable detailed logging. |
| `max_text_length` | `int` | `100000` | Maximum text length (security). |
| `enable_deduplication` | `bool` | `True` | Enable request deduplication. |
| `enable_metrics` | `bool` | `True` | Enable Prometheus metrics collection. |
| `enable_distributed_tracing` | `bool` | `True` | Enable request ID tracking. |

#### Optional Parameters - Autonomous Features

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_autonomous` | `bool` | `False` | Enable autonomous agent features. |
| `autonomous_features` | `Dict[str, Any]` | `None` | Configuration for autonomous features. |

##### autonomous_features Configuration

```python
autonomous_features = {
    "working_memory_ttl": 3600,        # TTL for working memory sessions (seconds)
    "goal_tracking_enabled": True,      # Enable goal tracking
    "metacognition_enabled": True,      # Enable quality/uncertainty assessment
    "confidence_threshold": 0.7         # Min confidence for quality assessment
}
```

---

## Core Methods

### load_memory_variables

Load memory variables from RecallBricks.

```python
def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]
```

**Parameters:**
- `inputs`: Dictionary containing the query (uses `input_key` or `"input"`)

**Returns:**
- Dictionary with `"history"` key containing memories

**Example:**
```python
result = memory.load_memory_variables({"input": "What are user preferences?"})
history = result["history"]
```

### save_context

Save context to RecallBricks with automatic metadata extraction.

```python
def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None
```

**Parameters:**
- `inputs`: Input dictionary
- `outputs`: Output dictionary

**Example:**
```python
memory.save_context(
    {"input": "What's the weather like?"},
    {"output": "It's sunny and 72 degrees."}
)
```

### learn

Save a memory with automatic metadata extraction.

```python
def learn(
    self,
    text: str,
    source: str = None,
    project_id: str = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]
```

**Parameters:**
- `text`: Text content to save
- `source`: Source identifier (defaults to instance source)
- `project_id`: Project ID (defaults to instance project_id)
- `metadata`: Optional additional metadata

**Returns:**
- Response from learn endpoint including extracted metadata

**Example:**
```python
result = memory.learn("User prefers dark mode for all applications")
print(result["metadata"]["tags"])  # ["preferences", "ui"]
```

### recall

Recall memories with optional organization.

```python
def recall(
    self,
    query: str,
    limit: int = None,
    organized: bool = None,
    project_id: str = None
) -> Dict[str, Any]
```

**Parameters:**
- `query`: Search query
- `limit`: Number of results (defaults to instance limit)
- `organized`: Use organized recall (defaults to instance setting)
- `project_id`: Project ID (defaults to instance project_id)

**Returns:**
- Response with memories and optional category summaries

**Example:**
```python
result = memory.recall("user preferences", limit=5, organized=True)
for mem in result["memories"]:
    print(mem["text"])
```

### clear

Clear memory (no-op for LangChain compatibility).

```python
def clear(self) -> None
```

---

## Monitoring Methods

### get_metrics

Get current metrics for monitoring.

```python
def get_metrics(self) -> Dict[str, int]
```

**Returns:**
```python
{
    "save_count": 0,
    "load_count": 0,
    "error_count": 0,
    "retry_count": 0
}
```

### get_detailed_metrics

Get detailed metrics including percentiles and rates.

```python
def get_detailed_metrics(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "requests_total": 100,
    "requests_success": 98,
    "requests_failed": 2,
    "response_time_avg": 0.15,
    "response_time_p50": 0.12,
    "response_time_p95": 0.25,
    "response_time_p99": 0.45,
    "success_rate": 0.98
}
```

### get_prometheus_metrics

Export metrics in Prometheus format.

```python
def get_prometheus_metrics(self) -> str
```

**Returns:**
```
# TYPE recallbricks_requests_total gauge
recallbricks_requests_total 100
# TYPE recallbricks_requests_success gauge
recallbricks_requests_success 98
...
```

### health_check

Comprehensive health check for monitoring and alerting.

```python
def health_check(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "status": "healthy",  # or "degraded", "unhealthy"
    "timestamp": "2024-01-15T10:30:00Z",
    "checks": {
        "api": {"status": "healthy"},
        "circuit_breaker": {"status": "healthy", "state": "closed"},
        "rate_limiter": {"status": "healthy"},
        "requests": {"status": "healthy", "success_rate": 0.98}
    }
}
```

### get_circuit_breaker_status

Get circuit breaker status for monitoring.

```python
def get_circuit_breaker_status(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "state": "closed",  # or "open", "half_open"
    "failure_count": 0,
    "last_failure_time": None
}
```

### reset_metrics

Reset all metrics to zero.

```python
def reset_metrics(self) -> None
```

### shutdown

Graceful shutdown - cleanup resources.

```python
def shutdown(self) -> None
```

---

## Autonomous Methods

See [Autonomous Features](./autonomous-features.md) for detailed documentation of:

- `create_working_memory_session(session_id)`
- `add_to_working_memory(session_id, item, metadata)`
- `get_working_memory(session_id)`
- `end_working_memory_session(session_id, persist)`
- `track_goal(goal_id, steps)`
- `complete_goal_step(goal_id, step_index)`
- `get_goal_status(goal_id)`
- `assess_quality(response, confidence)`
- `quantify_uncertainty(response, confidence, evidence)`
- `with_working_memory(session_id)` - Context manager
- `with_goal_tracking(goal_id, steps)` - Decorator
- `get_autonomous_status()`

---

## Properties

### memory_variables

Return memory variable names.

```python
@property
def memory_variables(self) -> List[str]
```

**Returns:** `["history"]`

---

## Exceptions

```python
from recallbricks_langchain import (
    RecallBricksError,      # Base exception
    ValidationError,         # Input validation failed
    RateLimitError,          # Rate limit exceeded
    CircuitBreakerError,     # Circuit breaker open
    APIError,                # API returned error
    TimeoutError,            # Request timed out
    DeduplicationError       # Duplicate request detected
)
```

### RecallBricksError

Base exception for all RecallBricks errors.

```python
class RecallBricksError(Exception):
    request_id: str = None
    metadata: dict
```

### APIError

Raised when API returns an error.

```python
class APIError(RecallBricksError):
    status_code: int = None
```

---

## Thread Safety

All `RecallBricksMemory` methods are thread-safe. The class uses:
- Internal locks for state management
- Thread-safe metrics collection
- Connection pooling with thread-safe session management

---

## Performance Considerations

1. **Connection Pooling**: Enabled by default, reuses TCP connections
2. **Request Deduplication**: Prevents duplicate saves within 60 seconds
3. **Circuit Breaker**: Opens after 5 failures, recovers after 60 seconds
4. **Rate Limiting**: 100 requests per 60 seconds by default
5. **Organized Recall**: 3-5x faster context assembly for LLMs
