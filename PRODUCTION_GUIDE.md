# Production Deployment Guide

## Quick Start for Production

### 1. Installation

```bash
pip install recallbricks-langchain
```

### 2. Basic Production Setup

```python
import os
import logging
from recallbricks_langchain import RecallBricksMemory
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI

# Configure logging for production
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recallbricks.log'),
        logging.StreamHandler()
    ]
)

def get_memory_for_user(user_id: str) -> RecallBricksMemory:
    """
    Factory function to create production-ready memory instance.
    """
    return RecallBricksMemory(
        api_key=os.getenv("RECALLBRICKS_API_KEY"),
        user_id=user_id,

        # Production tuning
        limit=20,
        min_relevance=0.7,
        max_retries=5,
        retry_delay=2.0,
        circuit_breaker_threshold=10,
        circuit_breaker_timeout=120,

        # Security
        max_text_length=50000,

        # Monitoring
        enable_logging=True
    )

# Use in your application
def handle_user_message(user_id: str, message: str):
    memory = get_memory_for_user(user_id)

    llm = ChatOpenAI(temperature=0.7)
    conversation = ConversationChain(llm=llm, memory=memory)

    response = conversation.run(message)
    return response
```

---

## Monitoring & Alerting

### Health Check Endpoint

```python
from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

# Global metrics collection
memory_instances = {}

@app.route('/health/recallbricks')
def recallbricks_health():
    """
    Health check endpoint for RecallBricks memory.
    Returns metrics and circuit breaker status.
    """
    if not memory_instances:
        return jsonify({"status": "no_instances"}), 200

    # Aggregate metrics across all instances
    total_metrics = {
        "save_count": 0,
        "load_count": 0,
        "error_count": 0,
        "retry_count": 0
    }

    circuit_breaker_states = {}

    for user_id, memory in memory_instances.items():
        metrics = memory.get_metrics()
        cb_status = memory.get_circuit_breaker_status()

        for key in total_metrics:
            total_metrics[key] += metrics[key]

        if cb_status["state"] != "closed":
            circuit_breaker_states[user_id] = cb_status

    # Calculate health
    total_ops = total_metrics["save_count"] + total_metrics["load_count"]
    error_rate = total_metrics["error_count"] / max(total_ops, 1)

    health_status = "healthy"
    if error_rate > 0.05:  # 5% error rate
        health_status = "degraded"
    if circuit_breaker_states:
        health_status = "critical"

    return jsonify({
        "status": health_status,
        "timestamp": datetime.now().isoformat(),
        "metrics": total_metrics,
        "error_rate": f"{error_rate:.2%}",
        "circuit_breakers_open": len(circuit_breaker_states),
        "circuit_breaker_details": circuit_breaker_states
    }), 200 if health_status == "healthy" else 503
```

### Prometheus Metrics Export

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
recallbricks_saves = Counter(
    'recallbricks_saves_total',
    'Total number of context saves',
    ['user_id']
)

recallbricks_loads = Counter(
    'recallbricks_loads_total',
    'Total number of memory loads',
    ['user_id']
)

recallbricks_errors = Counter(
    'recallbricks_errors_total',
    'Total number of errors',
    ['error_type']
)

recallbricks_latency = Histogram(
    'recallbricks_operation_duration_seconds',
    'Operation latency',
    ['operation']
)

circuit_breaker_state = Gauge(
    'recallbricks_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half_open)',
    ['user_id']
)

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

# Instrument your memory operations
class MonitoredRecallBricksMemory(RecallBricksMemory):
    """Wrapper that exports Prometheus metrics."""

    def save_context(self, inputs, outputs):
        with recallbricks_latency.labels(operation='save').time():
            try:
                super().save_context(inputs, outputs)
                recallbricks_saves.labels(user_id=self.user_id).inc()
            except Exception as e:
                recallbricks_errors.labels(error_type=type(e).__name__).inc()
                raise

    def load_memory_variables(self, inputs):
        with recallbricks_latency.labels(operation='load').time():
            try:
                result = super().load_memory_variables(inputs)
                recallbricks_loads.labels(user_id=self.user_id).inc()
                return result
            except Exception as e:
                recallbricks_errors.labels(error_type=type(e).__name__).inc()
                raise
```

### CloudWatch Integration (AWS)

```python
import boto3
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

def publish_memory_metrics(memory: RecallBricksMemory, namespace='RecallBricks'):
    """Publish metrics to AWS CloudWatch."""
    metrics = memory.get_metrics()
    cb_status = memory.get_circuit_breaker_status()

    # Calculate derived metrics
    total_ops = metrics["save_count"] + metrics["load_count"]
    error_rate = metrics["error_count"] / max(total_ops, 1) * 100

    # Publish to CloudWatch
    cloudwatch.put_metric_data(
        Namespace=namespace,
        MetricData=[
            {
                'MetricName': 'SaveOperations',
                'Value': metrics["save_count"],
                'Unit': 'Count',
                'Timestamp': datetime.now()
            },
            {
                'MetricName': 'LoadOperations',
                'Value': metrics["load_count"],
                'Unit': 'Count',
                'Timestamp': datetime.now()
            },
            {
                'MetricName': 'ErrorRate',
                'Value': error_rate,
                'Unit': 'Percent',
                'Timestamp': datetime.now()
            },
            {
                'MetricName': 'RetryCount',
                'Value': metrics["retry_count"],
                'Unit': 'Count',
                'Timestamp': datetime.now()
            },
            {
                'MetricName': 'CircuitBreakerOpen',
                'Value': 1 if cb_status["state"] == "open" else 0,
                'Unit': 'None',
                'Timestamp': datetime.now()
            }
        ]
    )

# Run periodically (e.g., every minute)
import schedule

def job():
    for user_id, memory in memory_instances.items():
        publish_memory_metrics(memory)

schedule.every(1).minutes.do(job)
```

---

## Error Handling Best Practices

### Graceful Degradation

```python
def get_conversation_with_fallback(user_id: str):
    """
    Create conversation with fallback to buffer memory if RecallBricks fails.
    """
    from langchain.memory import ConversationBufferMemory

    try:
        # Try RecallBricks first
        memory = RecallBricksMemory(
            api_key=os.getenv("RECALLBRICKS_API_KEY"),
            user_id=user_id,
            max_retries=3
        )

        # Test connection
        memory.load_memory_variables({"input": "test"})

        return memory, "recallbricks"

    except Exception as e:
        logging.error(f"RecallBricks failed, falling back to buffer: {e}")

        # Fallback to in-memory buffer
        memory = ConversationBufferMemory()
        return memory, "buffer"

# Use it
memory, memory_type = get_conversation_with_fallback(user_id)
logging.info(f"Using {memory_type} memory for user {user_id}")
```

### Circuit Breaker Alerts

```python
import smtplib
from email.message import EmailMessage

def check_circuit_breaker_and_alert(memory: RecallBricksMemory):
    """
    Check circuit breaker state and send alert if open.
    """
    cb_status = memory.get_circuit_breaker_status()

    if cb_status["state"] == "open":
        send_alert(
            subject="🚨 RecallBricks Circuit Breaker OPEN",
            message=f"""
            Circuit breaker has opened for RecallBricks memory.

            Details:
            - State: {cb_status['state']}
            - Failure count: {cb_status['failure_count']}
            - Last failure: {cb_status['last_failure_time']}

            Actions:
            1. Check RecallBricks API status
            2. Review recent error logs
            3. Consider scaling or failover
            """
        )

def send_alert(subject: str, message: str):
    """Send email alert."""
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = 'alerts@yourcompany.com'
    msg['To'] = 'oncall@yourcompany.com'

    # Send via SMTP
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.starttls()
        smtp.send_message(msg)
```

---

## Performance Optimization

### Connection Pooling (Advanced)

```python
from queue import Queue
import threading

class RecallBricksMemoryPool:
    """
    Connection pool for RecallBricks memory instances.
    Reduces overhead of creating new instances.
    """

    def __init__(
        self,
        api_key: str,
        pool_size: int = 100,
        **memory_kwargs
    ):
        self.api_key = api_key
        self.memory_kwargs = memory_kwargs
        self.pool = Queue(maxsize=pool_size)
        self._lock = threading.Lock()

        # Pre-populate pool
        for _ in range(pool_size):
            self.pool.put(self._create_memory())

    def _create_memory(self):
        """Create a new memory instance."""
        return RecallBricksMemory(
            api_key=self.api_key,
            **self.memory_kwargs
        )

    def acquire(self, user_id: str) -> RecallBricksMemory:
        """Get a memory instance from pool."""
        memory = self.pool.get()
        memory.user_id = user_id  # Set user context
        return memory

    def release(self, memory: RecallBricksMemory):
        """Return memory instance to pool."""
        memory.reset_metrics()
        self.pool.put(memory)

# Usage
pool = RecallBricksMemoryPool(
    api_key=os.getenv("RECALLBRICKS_API_KEY"),
    pool_size=100,
    max_retries=5
)

def handle_request(user_id: str, message: str):
    memory = pool.acquire(user_id)
    try:
        # Use memory
        conversation = ConversationChain(llm=llm, memory=memory)
        response = conversation.run(message)
        return response
    finally:
        pool.release(memory)
```

### Async Support (Future)

```python
# Coming soon: Async version
import asyncio
from typing import Dict, Any

class AsyncRecallBricksMemory(RecallBricksMemory):
    """Async version for high-concurrency applications."""

    async def load_memory_variables_async(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Async version of load_memory_variables."""
        # Implementation using aiohttp or httpx
        pass

    async def save_context_async(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """Async version of save_context."""
        pass

# Usage with FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.post("/chat")
async def chat(user_id: str, message: str):
    memory = AsyncRecallBricksMemory(api_key=os.getenv("RECALLBRICKS_API_KEY"))
    result = await memory.load_memory_variables_async({"input": message})
    return result
```

---

## Security Best Practices

### 1. API Key Management

```python
# ❌ BAD - Never hardcode
memory = RecallBricksMemory(api_key="rb_1234567890")

# ✅ GOOD - Use environment variables
import os
memory = RecallBricksMemory(api_key=os.getenv("RECALLBRICKS_API_KEY"))

# ✅ BETTER - Use secrets manager (AWS)
import boto3

def get_api_key():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='recallbricks/api-key')
    return response['SecretString']

memory = RecallBricksMemory(api_key=get_api_key())
```

### 2. User Authentication

```python
from functools import wraps
from flask import request, abort

def require_auth(f):
    """Ensure user is authenticated before accessing memory."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Verify JWT token, session, etc.
        user_id = verify_token(request.headers.get('Authorization'))
        if not user_id:
            abort(401)

        return f(user_id=user_id, *args, **kwargs)

    return decorated_function

@app.post('/chat')
@require_auth
def chat(user_id: str):
    # user_id is verified and trusted
    memory = get_memory_for_user(user_id)
    # ... rest of logic
```

### 3. Rate Limiting Per User

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.post('/chat')
@limiter.limit("10 per minute")  # Per-user rate limit
@require_auth
def chat(user_id: str, message: str):
    memory = get_memory_for_user(user_id)
    # ... handle chat
```

---

## Scaling Strategies

### Horizontal Scaling

```
                    Load Balancer
                         |
        +----------------+----------------+
        |                |                |
    Instance 1       Instance 2       Instance 3
        |                |                |
        +----------------+----------------+
                         |
                 RecallBricks API
```

**Benefits:**
- Each instance is stateless
- Linear scaling with user count
- No shared state to manage

**Configuration:**
```yaml
# docker-compose.yml
version: '3'
services:
  app:
    image: your-app:latest
    environment:
      - RECALLBRICKS_API_KEY=${RECALLBRICKS_API_KEY}
    deploy:
      replicas: 5  # Scale to 5 instances
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recallbricks-app
spec:
  replicas: 5
  selector:
    matchLabels:
      app: recallbricks-app
  template:
    metadata:
      labels:
        app: recallbricks-app
    spec:
      containers:
      - name: app
        image: your-app:latest
        env:
        - name: RECALLBRICKS_API_KEY
          valueFrom:
            secretKeyRef:
              name: recallbricks-secret
              key: api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: recallbricks-service
spec:
  selector:
    app: recallbricks-app
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

---

## Troubleshooting

### Common Issues

#### 1. Circuit Breaker Opens Frequently

**Symptoms:**
- `Circuit breaker is OPEN` errors
- High error rate

**Diagnosis:**
```python
cb_status = memory.get_circuit_breaker_status()
print(f"State: {cb_status['state']}")
print(f"Failures: {cb_status['failure_count']}")
print(f"Last failure: {cb_status['last_failure_time']}")
```

**Solutions:**
1. Check RecallBricks API status
2. Increase `circuit_breaker_threshold` if transient issues
3. Increase `circuit_breaker_timeout` for longer recovery
4. Check network connectivity

#### 2. High Latency

**Symptoms:**
- Slow response times
- Timeouts

**Diagnosis:**
```python
import time

start = time.time()
result = memory.load_memory_variables({"input": "test"})
latency = time.time() - start
print(f"Latency: {latency:.2f}s")
```

**Solutions:**
1. Reduce `limit` parameter (fetch less memories)
2. Increase `min_relevance` (more selective)
3. Check RecallBricks API latency
4. Consider caching frequently accessed memories

#### 3. Memory Leaks

**Symptoms:**
- Increasing memory usage over time
- OOM errors

**Diagnosis:**
```python
import psutil
import gc

process = psutil.Process()

for i in range(1000):
    memory = RecallBricksMemory(api_key=api_key, user_id=f"user-{i}")
    memory.save_context({"input": "test"}, {"output": "test"})
    del memory

    if i % 100 == 0:
        gc.collect()
        mem = process.memory_info().rss / 1024 / 1024
        print(f"Iteration {i}: {mem:.2f} MB")
```

**Solutions:**
1. Ensure proper cleanup (`del memory`)
2. Use connection pooling
3. Monitor with `get_metrics()`

---

## Launch Checklist

### Before Going Live

- [ ] Environment variables configured
- [ ] API key from production RecallBricks account
- [ ] Logging configured and tested
- [ ] Monitoring/alerting set up
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Backup/fallback strategy in place
- [ ] Documentation updated
- [ ] Team trained on operations

### Day 1 Operations

- [ ] Monitor error rates (target < 1%)
- [ ] Watch circuit breaker state
- [ ] Track latency (P95, P99)
- [ ] Monitor memory usage
- [ ] Check logs for warnings
- [ ] Verify metrics collection
- [ ] Test alerts are working

### Week 1 Review

- [ ] Analyze performance trends
- [ ] Review error patterns
- [ ] Optimize configuration if needed
- [ ] Gather user feedback
- [ ] Document any issues/resolutions
- [ ] Plan improvements

---

## Support & Resources

### Getting Help

1. **RecallBricks Support:** support@recallbricks.com
2. **GitHub Issues:** https://github.com/recallbricks/recallbricks-langchain/issues
3. **Documentation:** https://recallbricks.com/docs

### Useful Links

- [RecallBricks API Docs](https://recallbricks.com/docs#api-reference)
- [LangChain Documentation](https://python.langchain.com/docs)
- [Enterprise Validation Report](./ENTERPRISE_VALIDATION.md)

---

**Last Updated:** November 16, 2025
**Version:** 0.1.0
