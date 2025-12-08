"""RecallBricks LangChain Integration

Official LangChain integration for RecallBricks Memory Graph.

Enterprise-grade features:
- Automatic retry with exponential backoff
- Circuit breaker for fault tolerance
- Rate limiting to prevent API abuse
- Request deduplication
- Distributed tracing
- Prometheus metrics export
- Health checks
- Graceful shutdown
"""

from recallbricks_langchain.memory import (
    RecallBricksMemory,
    # Custom exceptions
    RecallBricksError,
    ValidationError,
    RateLimitError,
    CircuitBreakerError,
    APIError,
    TimeoutError,
    DeduplicationError,
)

__version__ = "0.2.0"  # Bumped for bulletproof features
__all__ = [
    "RecallBricksMemory",
    "RecallBricksError",
    "ValidationError",
    "RateLimitError",
    "CircuitBreakerError",
    "APIError",
    "TimeoutError",
    "DeduplicationError",
]
