from typing import Any, Dict, List, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import requests
import os
import time
import logging
import threading
import uuid
import json
import hashlib
import atexit
from functools import wraps
from datetime import datetime, timedelta, timezone
from collections import deque


# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS - Enterprise-grade error handling
# ============================================================================

class RecallBricksError(Exception):
    """Base exception for all RecallBricks errors."""
    def __init__(self, message: str, request_id: str = None, **kwargs):
        self.request_id = request_id
        self.metadata = kwargs
        super().__init__(message)


class ValidationError(RecallBricksError):
    """Raised when input validation fails."""
    pass


class RateLimitError(RecallBricksError):
    """Raised when rate limit is exceeded."""
    pass


class CircuitBreakerError(RecallBricksError):
    """Raised when circuit breaker is open."""
    pass


class APIError(RecallBricksError):
    """Raised when API returns an error."""
    def __init__(self, message: str, status_code: int = None, **kwargs):
        self.status_code = status_code
        super().__init__(message, **kwargs)


class TimeoutError(RecallBricksError):
    """Raised when request times out."""
    pass


class DeduplicationError(RecallBricksError):
    """Raised when duplicate request is detected."""
    pass


# ============================================================================
# SHARED SESSION - Connection pooling
# ============================================================================

# Shared requests session for connection pooling
_session = None
_session_lock = threading.Lock()


def get_session():
    """
    Get shared requests session with connection pooling.
    PERFORMANCE FIX: Reuses TCP connections across requests.
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=100,
                    max_retries=0  # We handle retries ourselves
                )
                _session.mount('https://', adapter)
                _session.mount('http://', adapter)
    return _session


class CircuitBreaker:
    """
    Circuit breaker pattern for handling API failures gracefully.
    Prevents cascading failures by temporarily blocking requests after failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        # SECURITY FIX: Hold lock for entire state check to prevent race condition
        with self._lock:
            if self.state == "open":
                if datetime.now(timezone.utc) - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                    self.state = "half_open"
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise Exception("Circuit breaker is OPEN - too many failures")

            # Capture state while holding lock
            state_before_call = self.state

        # Execute function outside lock to avoid blocking other threads
        try:
            result = func(*args, **kwargs)

            with self._lock:
                if state_before_call == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed - service recovered")

            return result

        except self.expected_exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now(timezone.utc)

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")

            raise e


class RateLimiter:
    """
    Token bucket rate limiter for API request throttling.
    SECURITY FIX: Prevents API abuse and cost attacks.
    """

    def __init__(self, rate: int = 100, per: int = 60):
        """
        Initialize rate limiter.

        Args:
            rate: Maximum number of requests
            per: Per this many seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = float(rate)
        self.last_check = time.time()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """
        Check if request is allowed under rate limit.

        Returns:
            True if allowed, False if rate limit exceeded
        """
        with self._lock:
            current = time.time()
            elapsed = current - self.last_check
            self.last_check = current

            # Add tokens based on time elapsed
            self.allowance += elapsed * (self.rate / self.per)

            # Cap at maximum rate
            if self.allowance > self.rate:
                self.allowance = self.rate

            # Check if we have tokens available
            if self.allowance < 1.0:
                return False

            # Consume one token
            self.allowance -= 1.0
            return True

    def wait_if_needed(self):
        """Wait if rate limit exceeded, then allow request."""
        while not self.allow():
            time.sleep(0.1)


class RequestDeduplicator:
    """
    Request deduplication to prevent double-saves.
    Uses content hashing with sliding window.
    """

    def __init__(self, window_size: int = 1000, window_seconds: int = 60):
        """
        Initialize deduplicator.

        Args:
            window_size: Maximum number of requests to track
            window_seconds: Time window for deduplication
        """
        self.window_size = window_size
        self.window_seconds = window_seconds
        self.recent_requests = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def _hash_request(self, data: Dict[str, Any]) -> str:
        """Generate hash of request data."""
        # Create stable hash from request data
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def is_duplicate(self, data: Dict[str, Any]) -> bool:
        """
        Check if request is a duplicate.

        Args:
            data: Request data to check

        Returns:
            True if duplicate, False otherwise
        """
        request_hash = self._hash_request(data)
        current_time = time.time()

        with self._lock:
            # Clean old entries
            cutoff_time = current_time - self.window_seconds
            while self.recent_requests and self.recent_requests[0][1] < cutoff_time:
                self.recent_requests.popleft()

            # Check for duplicate
            for stored_hash, timestamp in self.recent_requests:
                if stored_hash == request_hash:
                    return True

            # Add new request
            self.recent_requests.append((request_hash, current_time))
            return False


class MetricsCollector:
    """
    Prometheus-compatible metrics collector.
    Thread-safe metrics aggregation.
    """

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "requests_rate_limited": 0,
            "requests_deduplicated": 0,
            "circuit_breaker_opened": 0,
            "circuit_breaker_closed": 0,
            "response_time_sum": 0.0,
            "response_time_count": 0,
            "retries_total": 0,
        }
        self.response_times = deque(maxlen=1000)  # Last 1000 response times
        self._lock = threading.Lock()

    def increment(self, metric: str, value: int = 1):
        """Increment a counter metric."""
        with self._lock:
            if metric in self.metrics:
                self.metrics[metric] += value

    def record_response_time(self, duration: float):
        """Record response time for percentile calculations."""
        with self._lock:
            self.metrics["response_time_sum"] += duration
            self.metrics["response_time_count"] += 1
            self.response_times.append(duration)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            metrics = self.metrics.copy()

            # Calculate percentiles
            if self.response_times:
                sorted_times = sorted(self.response_times)
                metrics["response_time_p50"] = sorted_times[len(sorted_times) // 2]
                metrics["response_time_p95"] = sorted_times[int(len(sorted_times) * 0.95)]
                metrics["response_time_p99"] = sorted_times[int(len(sorted_times) * 0.99)]

            # Calculate averages
            if metrics["response_time_count"] > 0:
                metrics["response_time_avg"] = (
                    metrics["response_time_sum"] / metrics["response_time_count"]
                )

            # Calculate success rate
            if metrics["requests_total"] > 0:
                metrics["success_rate"] = (
                    metrics["requests_success"] / metrics["requests_total"]
                )

            return metrics

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.get_metrics()
        lines = []

        for key, value in metrics.items():
            # Convert to Prometheus format
            metric_name = f"recallbricks_{key}"
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")

        return "\n".join(lines)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying operations with exponential backoff.
    Essential for handling transient API failures in production.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1

                    if retries >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise

                    # Calculate exponential backoff with jitter
                    import random
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = min(delay + jitter, max_delay)

                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} "
                        f"after {sleep_time:.2f}s delay. Error: {e}"
                    )

                    time.sleep(sleep_time)
                    delay *= exponential_base

            return func(*args, **kwargs)

        return wrapper
    return decorator


class RecallBricksMemory:
    """
    Enterprise-grade memory class for LangChain using RecallBricks.

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Comprehensive error handling
    - Thread-safe operations
    - Logging and monitoring support
    - Input validation and sanitization
    """

    def __init__(
        self,
        agent_id: str,
        user_id: str = None,
        service_token: str = None,
        api_url: str = "https://recallbricks-api-clean.onrender.com",
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
        enable_distributed_tracing: bool = True
    ):
        """
        Initialize enterprise-grade RecallBricks memory.

        Args:
            agent_id: RecallBricks agent ID (required)
            user_id: Optional user ID for multi-user applications
            service_token: RecallBricks service token (optional, defaults to RECALLBRICKS_SERVICE_TOKEN env var)
            api_url: RecallBricks API base URL (defaults to production)
            return_messages: Whether to return messages or string
            input_key: Key to use for input in chat history
            output_key: Key to use for output in chat history
            limit: Max number of memories to return
            min_relevance: Minimum relevance score (0-1)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay for exponential backoff (seconds)
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Seconds before attempting recovery
            enable_logging: Enable detailed logging
            max_text_length: Maximum allowed text length (security)
            rate_limit: Maximum requests per period (default: 100)
            rate_limit_period: Period in seconds (default: 60)
            enable_deduplication: Enable request deduplication (default: True)
            enable_metrics: Enable Prometheus metrics collection (default: True)
            enable_distributed_tracing: Enable request ID tracking (default: True)
        """
        # Store memory configuration
        self.return_messages = return_messages
        self.input_key = input_key
        self.output_key = output_key

        # SECURITY FIX: Validate agent_id
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")

        # SECURITY FIX: Validate HTTPS only
        if not api_url.startswith('https://'):
            raise ValueError(
                f"api_url must use HTTPS for security. Got: {api_url}"
            )

        # SECURITY FIX: Validate user_id is UUID format if provided
        if user_id is not None:
            try:
                uuid.UUID(user_id)
            except (ValueError, AttributeError, TypeError):
                raise ValueError(
                    f"user_id must be a valid UUID format. Got: {user_id}. "
                    f"Generate with: str(uuid.uuid4())"
                )

        # Get service token from parameter or environment variable
        self.service_token = service_token or os.getenv("RECALLBRICKS_SERVICE_TOKEN")
        if not self.service_token:
            raise ValueError(
                "service_token must be provided or RECALLBRICKS_SERVICE_TOKEN environment variable must be set"
            )

        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")

        if not 0 <= min_relevance <= 1:
            raise ValueError("min_relevance must be between 0 and 1")

        # Initialize API configuration
        self.agent_id = agent_id
        self.api_url = api_url.rstrip('/')
        self.user_id = user_id
        self.limit = limit
        self.min_relevance = min_relevance
        self.max_text_length = max_text_length

        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_timeout
        )

        # SECURITY FIX: Rate limiter to prevent API abuse
        self.rate_limiter = RateLimiter(
            rate=rate_limit,
            per=rate_limit_period
        )

        # BULLETPROOF: Request deduplication
        self.enable_deduplication = enable_deduplication
        if enable_deduplication:
            self.deduplicator = RequestDeduplicator()

        # BULLETPROOF: Metrics collection
        self.enable_metrics = enable_metrics
        if enable_metrics:
            self.metrics_collector = MetricsCollector()

        # BULLETPROOF: Distributed tracing
        self.enable_distributed_tracing = enable_distributed_tracing

        # Logging
        self.enable_logging = enable_logging
        if enable_logging:
            # Mask user_id for privacy (only log first 8 chars)
            masked_user = user_id[:8] + "..." if user_id and len(user_id) > 8 else user_id
            logger.info(
                f"RecallBricksMemory initialized for agent: {agent_id}, user: {masked_user or 'default'}"
            )

        # Thread safety
        self._lock = threading.Lock()

        # Legacy metrics (kept for backwards compatibility)
        self.metrics = {
            "save_count": 0,
            "load_count": 0,
            "error_count": 0,
            "retry_count": 0
        }

        # BULLETPROOF: Graceful shutdown
        self._shutdown = False
        atexit.register(self.shutdown)

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return ["history"]

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize input text for security and size limits.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text

        Raises:
            ValueError: If text exceeds maximum length
        """
        if not isinstance(text, str):
            text = str(text)

        # Remove null bytes (security)
        text = text.replace('\x00', '')

        # Check length
        if len(text) > self.max_text_length:
            logger.warning(
                f"Text length ({len(text)}) exceeds maximum ({self.max_text_length}). Truncating."
            )
            text = text[:self.max_text_length]

        return text

    def _execute_with_retry(self, func: Callable, *args, **kwargs):
        """
        Execute a function with retry logic and circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail or circuit breaker is open
        """
        # BULLETPROOF: Generate request ID for tracing
        request_id = self._generate_request_id() if self.enable_distributed_tracing else None

        # BULLETPROOF: Track request start time
        start_time = time.time()

        # BULLETPROOF: Check shutdown status
        if self._shutdown:
            raise RecallBricksError("Service is shutting down", request_id=request_id)

        # BULLETPROOF: Increment total requests
        if self.enable_metrics:
            self.metrics_collector.increment("requests_total")

        # SECURITY FIX: Check rate limit before making request
        if not self.rate_limiter.allow():
            if self.enable_metrics:
                self.metrics_collector.increment("requests_rate_limited")
            raise RateLimitError("Rate limit exceeded. Please slow down requests.", request_id=request_id)

        retries = 0
        delay = self.retry_delay
        last_exception = None

        while retries <= self.max_retries:
            try:
                # Use circuit breaker
                result = self.circuit_breaker.call(func, *args, **kwargs)

                # BULLETPROOF: Record success metrics
                if self.enable_metrics:
                    self.metrics_collector.increment("requests_success")
                    duration = time.time() - start_time
                    self.metrics_collector.record_response_time(duration)

                return result

            except Exception as e:
                last_exception = e
                retries += 1

                with self._lock:
                    self.metrics["retry_count"] += 1

                # BULLETPROOF: Track retry metrics
                if self.enable_metrics:
                    self.metrics_collector.increment("retries_total")

                if retries > self.max_retries:
                    with self._lock:
                        self.metrics["error_count"] += 1

                    # BULLETPROOF: Track failure metrics
                    if self.enable_metrics:
                        self.metrics_collector.increment("requests_failed")

                    if self.enable_logging:
                        log_msg = f"[{request_id}] Failed after {self.max_retries} retries: {e}" if request_id else f"Failed after {self.max_retries} retries: {e}"
                        logger.error(log_msg)
                    raise

                # Exponential backoff with jitter
                import random
                jitter = random.uniform(0, 0.1 * delay)
                sleep_time = min(delay + jitter, 60.0)

                if self.enable_logging:
                    log_msg = f"[{request_id}] Retry {retries}/{self.max_retries} after {sleep_time:.2f}s. Error: {e}" if request_id else f"Retry {retries}/{self.max_retries} after {sleep_time:.2f}s. Error: {e}"
                    logger.warning(log_msg)

                time.sleep(sleep_time)
                delay *= 2

        raise last_exception

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables from RecallBricks with enterprise-grade reliability.

        Features:
        - Automatic retry with exponential backoff
        - Circuit breaker protection
        - Input validation and sanitization
        - Comprehensive error handling

        Args:
            inputs: Input dictionary containing the query

        Returns:
            Dictionary with memory history

        Raises:
            ValueError: If inputs are invalid
            Exception: If API calls fail after retries
        """
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a dictionary")

        query = inputs.get(self.input_key or "input", "")

        # Sanitize input
        try:
            query = self._sanitize_text(query)
        except Exception as e:
            logger.error(f"Failed to sanitize query: {e}")
            return {"history": [] if self.return_messages else ""}

        # Track metrics
        with self._lock:
            self.metrics["load_count"] += 1

        # Define API call operation
        def get_context_operation():
            # Use GET /api/v1/memories for immediate results
            # TODO: Switch to search endpoint once embedding indexing is faster
            url = f"{self.api_url}/api/v1/memories"
            headers = {
                "X-Service-Token": self.service_token,
                "Content-Type": "application/json"
            }
            params = {
                "user_id": self.user_id,
                "limit": self.limit
            }

            # PERFORMANCE FIX: Use connection pool
            response = get_session().get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()

        try:
            # Execute with retry and circuit breaker
            response_data = self._execute_with_retry(get_context_operation)

            # Extract memories from response
            memories = response_data.get("memories", [])

            # Sort by timestamp (most recent first) and apply limit
            memories = sorted(
                memories,
                key=lambda m: m.get('created_at', ''),
                reverse=True
            )[:self.limit]

            if self.enable_logging:
                logger.debug(
                    f"Retrieved {len(memories)} memories for user: {self.user_id or 'default'}"
                )

            # Format based on return_messages setting
            if self.return_messages:
                # Return memories as messages (simplified for now)
                # In a real implementation, you might want to parse the memory text
                # to determine if it's from Human or AI
                messages = []
                for memory in memories:
                    text = memory.get("text", "")
                    if text:
                        # For now, treat all memories as context (could be improved)
                        messages.append(HumanMessage(content=text))

                return {"history": messages}
            else:
                # Return as formatted string
                context_parts = []
                for memory in memories:
                    text = memory.get("text", "")
                    if text:
                        context_parts.append(text)

                context = "\n\n".join(context_parts)
                return {"history": context}

        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to load memory: {e}")

            # Return empty history on failure (graceful degradation)
            return {"history": [] if self.return_messages else ""}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context to RecallBricks with enterprise-grade reliability.

        Features:
        - Automatic retry with exponential backoff
        - Circuit breaker protection
        - Input validation and sanitization
        - Thread-safe metrics tracking

        Args:
            inputs: Input dictionary
            outputs: Output dictionary

        Raises:
            ValueError: If inputs/outputs are invalid
            Exception: If API calls fail after retries
        """
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError("inputs and outputs must be dictionaries")

        input_str = inputs.get(self.input_key or "input", "")
        output_str = outputs.get(self.output_key or "output", "")

        # Sanitize inputs
        try:
            input_str = self._sanitize_text(input_str)
            output_str = self._sanitize_text(output_str)
        except Exception as e:
            logger.error(f"Failed to sanitize context: {e}")
            raise ValueError(f"Invalid context data: {e}")

        # SECURITY FIX: Validate not empty or whitespace-only
        if not input_str or not input_str.strip():
            if self.enable_logging:
                logger.warning("Skipping save: empty or whitespace-only input")
            return

        if not output_str or not output_str.strip():
            if self.enable_logging:
                logger.warning("Skipping save: empty or whitespace-only output")
            return

        # BULLETPROOF: Check for duplicate request
        if self.enable_deduplication and hasattr(self, 'deduplicator'):
            dedup_data = {
                "input": input_str,
                "output": output_str,
                "user_id": self.user_id,
                "agent_id": self.agent_id
            }

            if self.deduplicator.is_duplicate(dedup_data):
                if self.enable_logging:
                    logger.warning("Skipping save: duplicate request detected")

                if self.enable_metrics:
                    self.metrics_collector.increment("requests_deduplicated")

                return  # Skip duplicate save

        # Format as conversation turn
        text = f"Human: {input_str}\nAI: {output_str}"

        # Track metrics
        with self._lock:
            self.metrics["save_count"] += 1

        # Define save operation
        def save_operation():
            url = f"{self.api_url}/api/v1/memories"
            headers = {
                "X-Service-Token": self.service_token,
                "Content-Type": "application/json"
            }
            payload = {
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "text": text,  # API expects 'text' field, not 'content'
                "metadata": {
                    "type": "conversation_turn",
                    "timestamp": datetime.now(timezone.utc).isoformat()  # SECURITY FIX: UTC timezone
                }
            }

            # SECURITY FIX: Validate total payload size
            payload_size = len(json.dumps(payload))
            if payload_size > 250000:  # 250KB limit (conservative)
                raise ValueError(f"Total payload too large: {payload_size} bytes (max 250KB)")

            # PERFORMANCE FIX: Use connection pool
            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()

        try:
            # Execute with retry and circuit breaker
            self._execute_with_retry(save_operation)

            if self.enable_logging:
                logger.debug(
                    f"Saved context for user: {self.user_id or 'default'}"
                )

        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to save context: {e}")
            raise

    def get_metrics(self) -> Dict[str, int]:
        """
        Get current metrics for monitoring.

        Returns:
            Dictionary with metric counts
        """
        with self._lock:
            return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        with self._lock:
            for key in self.metrics:
                self.metrics[key] = 0

        if self.enable_logging:
            logger.info("Metrics reset")

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Get circuit breaker status for monitoring.

        Returns:
            Dictionary with circuit breaker state
        """
        return {
            "state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "last_failure_time": self.circuit_breaker.last_failure_time
        }

    def _generate_request_id(self) -> str:
        """
        Generate unique request ID for distributed tracing.

        Returns:
            UUID string for request tracking
        """
        return str(uuid.uuid4())

    def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        if self.enable_metrics and hasattr(self, 'metrics_collector'):
            return self.metrics_collector.export_prometheus()
        return ""

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics including percentiles and rates.

        Returns:
            Comprehensive metrics dictionary
        """
        if self.enable_metrics and hasattr(self, 'metrics_collector'):
            return self.metrics_collector.get_metrics()
        return self.get_metrics()  # Fallback to legacy metrics

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for monitoring and alerting.

        Returns:
            Health status dictionary with detailed diagnostics
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }

        try:
            # Check circuit breaker
            cb_status = self.get_circuit_breaker_status()
            health["checks"]["circuit_breaker"] = {
                "status": "healthy" if cb_status["state"] == "closed" else "degraded",
                "state": cb_status["state"],
                "failure_count": cb_status["failure_count"]
            }

            # Check rate limiter
            health["checks"]["rate_limiter"] = {
                "status": "healthy",
                "allowance": self.rate_limiter.allowance,
                "rate": f"{self.rate_limiter.rate}/{self.rate_limiter.per}s"
            }

            # Check metrics
            if self.enable_metrics and hasattr(self, 'metrics_collector'):
                metrics = self.metrics_collector.get_metrics()
                success_rate = metrics.get("success_rate", 0)

                health["checks"]["requests"] = {
                    "status": "healthy" if success_rate > 0.95 else "degraded",
                    "success_rate": success_rate,
                    "total": metrics.get("requests_total", 0),
                    "avg_response_time": metrics.get("response_time_avg", 0)
                }

            # Check shutdown status
            health["checks"]["shutdown"] = {
                "status": "healthy" if not self._shutdown else "shutting_down",
                "shutdown_initiated": self._shutdown
            }

            # Overall status
            if any(check["status"] == "unhealthy" for check in health["checks"].values()):
                health["status"] = "unhealthy"
            elif any(check["status"] == "degraded" for check in health["checks"].values()):
                health["status"] = "degraded"

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health

    def shutdown(self):
        """
        Graceful shutdown - cleanup resources.
        Called automatically via atexit.
        """
        if self._shutdown:
            return

        self._shutdown = True

        if self.enable_logging:
            logger.info("Shutting down RecallBricksMemory gracefully...")

        # Log final metrics
        if self.enable_metrics and hasattr(self, 'metrics_collector'):
            final_metrics = self.metrics_collector.get_metrics()
            if self.enable_logging:
                logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")

        if self.enable_logging:
            logger.info("Shutdown complete")

    def clear(self) -> None:
        """
        Clear memory (optional implementation).

        Note: RecallBricks doesn't support bulk delete yet.
        This is a no-op but included for LangChain compatibility.
        """
        if self.enable_logging:
            logger.info("Clear called (no-op - RecallBricks doesn't support bulk delete)")
