from typing import Any, Dict, List, Optional, Callable, Generator
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
from contextlib import contextmanager


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
    - Automatic metadata extraction via learn() endpoint
    - Organized recall with category summaries (3-5x faster context assembly)
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
    ):
        """
        Initialize enterprise-grade RecallBricks memory.

        Args:
            agent_id: RecallBricks agent ID (required)
            user_id: Optional user ID for multi-user applications
            api_key: RecallBricks API key (optional, defaults to RECALLBRICKS_API_KEY env var)
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
            organized: Use organized recall for better context assembly (default: True)
            source: Source identifier for memories (default: "langchain")
            project_id: Optional project ID for multi-tenant applications
            enable_autonomous: Enable autonomous agent features (default: False)
            autonomous_features: Configuration dict for autonomous features:
                - working_memory_ttl: TTL for working memory sessions (seconds, default: 3600)
                - goal_tracking_enabled: Enable goal tracking (default: True)
                - metacognition_enabled: Enable quality/uncertainty assessment (default: True)
                - confidence_threshold: Min confidence for quality assessment (default: 0.7)
        """
        # Store memory configuration
        self.return_messages = return_messages
        self.input_key = input_key
        self.output_key = output_key
        self.organized = organized
        self.source = source
        self.project_id = project_id

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

        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("RECALLBRICKS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "api_key must be provided or RECALLBRICKS_API_KEY environment variable must be set"
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

        # Autonomous agent features (v1.3.0)
        self.enable_autonomous = enable_autonomous
        self.autonomous_features = autonomous_features or {}

        # Default autonomous configuration
        self._autonomous_config = {
            "working_memory_ttl": self.autonomous_features.get("working_memory_ttl", 3600),
            "goal_tracking_enabled": self.autonomous_features.get("goal_tracking_enabled", True),
            "metacognition_enabled": self.autonomous_features.get("metacognition_enabled", True),
            "confidence_threshold": self.autonomous_features.get("confidence_threshold", 0.7),
        }

        # Working memory sessions storage
        self._working_memory_sessions: Dict[str, Dict[str, Any]] = {}

        # Goal tracking storage
        self._active_goals: Dict[str, Dict[str, Any]] = {}

        if enable_autonomous and self.enable_logging:
            logger.info(f"Autonomous agent features enabled with config: {self._autonomous_config}")

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
        - Organized recall with category summaries (when organized=True)
        - 3-5x faster context assembly for LLMs
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

        # Get query from inputs, with fallback for empty queries
        query = inputs.get(self.input_key or "input", "")
        if not query or not query.strip():
            query = "conversation history"  # Default fallback for semantic search

        # Sanitize input
        try:
            query = self._sanitize_text(query)
        except Exception as e:
            logger.error(f"Failed to sanitize query: {e}")
            return {"history": []}

        # Track metrics
        with self._lock:
            self.metrics["load_count"] += 1

        # Define API call operation
        def get_context_operation():
            # Use POST /memories/recall for organized recall
            # User scoping handled server-side via X-API-Key
            url = f"{self.api_url}/memories/recall"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "query": query,
                "limit": self.limit,
                "organized": self.organized
            }

            # Add optional project_id if set
            if self.project_id:
                payload["project_id"] = self.project_id

            # PERFORMANCE FIX: Use connection pool
            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                # Debug logging for 400 errors
                logger.error(f"DEBUG URL: {url}")
                logger.error(f"DEBUG Headers: {dict(headers)}")
                logger.error(f"DEBUG Payload: {payload}")
                logger.error(f"DEBUG Response Status: {response.status_code}")
                logger.error(f"DEBUG Response Body: {response.text}")
                raise
            return response.json()

        try:
            # Execute with retry and circuit breaker
            response_data = self._execute_with_retry(get_context_operation)

            # Extract memories from response
            memories = response_data.get("memories", [])
            categories = response_data.get("categories", {})

            if self.enable_logging:
                logger.debug(
                    f"Retrieved {len(memories)} memories for user: {self.user_id or 'default'}"
                )

            # Format based on return_messages setting
            if self.return_messages:
                # Return memories as messages
                messages = []
                for memory in memories:
                    text = memory.get("text", "")
                    if text:
                        messages.append(HumanMessage(content=text))

                return {"history": messages}
            else:
                # Return as formatted string with organized context
                if self.organized and categories:
                    context = self._format_organized_memories(memories, categories)
                else:
                    # Backward compatible format
                    context_parts = []
                    for memory in memories:
                        text = memory.get("text", "")
                        if text:
                            context_parts.append(text)
                    context = "\n\n".join(context_parts)

                return {"history": [context] if context else []}

        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to load memory: {e}")

            # Return empty history on failure (graceful degradation)
            return {"history": []}

    def _format_organized_memories(self, memories: List[Dict], categories: Dict) -> str:
        """
        Format organized recall results for LangChain context.

        Args:
            memories: List of memory objects from recall
            categories: Category summaries from organized recall

        Returns:
            Formatted context string with category organization
        """
        if not memories:
            return ""

        context_parts = ["=== Relevant Context ===\n"]

        # Add category summaries
        if categories:
            context_parts.append("Overview:")
            for category, summary_data in categories.items():
                if isinstance(summary_data, dict):
                    summary = summary_data.get("summary", "")
                    count = summary_data.get("count", 0)
                    avg_score = summary_data.get("avg_score", 0)
                    context_parts.append(
                        f"  • {category}: {summary} "
                        f"({count} memories, relevance: {avg_score:.2f})"
                    )
            context_parts.append("")

        # Add individual memories grouped by category
        context_parts.append("Details:")
        current_category = None

        for memory in memories:
            metadata = memory.get("metadata", {})
            memory_category = metadata.get("category", "General")

            if memory_category != current_category:
                current_category = memory_category
                context_parts.append(f"\n[{current_category}]")

            text = memory.get("text", "")
            if text:
                context_parts.append(f"  - {text}")

        return "\n".join(context_parts)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context to RecallBricks with automatic metadata extraction.

        Features:
        - Uses learn() endpoint for automatic tag/category extraction
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

        # Track metrics
        with self._lock:
            self.metrics["save_count"] += 1

        # Save input with automatic metadata extraction via learn()
        self._learn(f"User: {input_str}")

        # Save output with automatic metadata extraction via learn()
        self._learn(f"Assistant: {output_str}")

        if self.enable_logging:
            logger.debug(
                f"Saved context for user: {self.user_id or 'default'}"
            )

    def _learn(self, text: str) -> Dict[str, Any]:
        """
        Save memory using learn() endpoint with automatic metadata extraction.

        Args:
            text: Text content to save

        Returns:
            Response from learn endpoint

        Raises:
            Exception: If API call fails after retries
        """
        def learn_operation():
            url = f"{self.api_url}/memories/learn"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "source": self.source
            }

            # Add optional fields if set
            if self.project_id:
                payload["project_id"] = self.project_id

            # SECURITY FIX: Validate total payload size
            payload_size = len(json.dumps(payload))
            if payload_size > 250000:  # 250KB limit (conservative)
                raise ValueError(f"Total payload too large: {payload_size} bytes (max 250KB)")

            # PERFORMANCE FIX: Use connection pool
            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()

        try:
            return self._execute_with_retry(learn_operation)
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to learn memory: {e}")
            raise

    def learn(self, text: str, source: str = None, project_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Public method to save a memory with automatic metadata extraction.

        This is the recommended way to save memories as it automatically
        extracts tags, categories, entities, and importance scores.

        Args:
            text: Text content to save
            source: Source identifier (defaults to instance source)
            project_id: Project ID (defaults to instance project_id)
            metadata: Optional additional metadata

        Returns:
            Response from learn endpoint including extracted metadata

        Example:
            memory.learn("User prefers dark mode for all applications")
            # Returns: {"id": "...", "metadata": {"tags": ["preferences", "ui"], ...}}
        """
        def learn_operation():
            url = f"{self.api_url}/memories/learn"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "text": self._sanitize_text(text),
                "source": source or self.source
            }

            # Add optional fields
            if project_id or self.project_id:
                payload["project_id"] = project_id or self.project_id
            if metadata:
                payload["metadata"] = metadata

            # SECURITY FIX: Validate total payload size
            payload_size = len(json.dumps(payload))
            if payload_size > 250000:  # 250KB limit (conservative)
                raise ValueError(f"Total payload too large: {payload_size} bytes (max 250KB)")

            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()

        return self._execute_with_retry(learn_operation)

    def recall(self, query: str, limit: int = None, organized: bool = None, project_id: str = None) -> Dict[str, Any]:
        """
        Public method to recall memories with optional organization.

        Args:
            query: Search query
            limit: Number of results (defaults to instance limit)
            organized: Use organized recall (defaults to instance setting)
            project_id: Project ID (defaults to instance project_id)

        Returns:
            Response with memories and optional category summaries

        Example:
            result = memory.recall("user preferences")
            # Returns: {"memories": [...], "categories": {"Preferences": {...}}}
        """
        def recall_operation():
            url = f"{self.api_url}/memories/recall"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "query": self._sanitize_text(query),
                "limit": limit or self.limit,
                "organized": organized if organized is not None else self.organized
            }

            if project_id or self.project_id:
                payload["project_id"] = project_id or self.project_id

            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()

        return self._execute_with_retry(recall_operation)

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
        Calls the RecallBricks API /health endpoint and checks internal state.

        Returns:
            Health status dictionary with detailed diagnostics
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }

        try:
            # Check API health endpoint
            try:
                url = f"{self.api_url}/health"
                headers = {"X-API-Key": self.api_key}
                response = get_session().get(url, headers=headers, timeout=10)
                response.raise_for_status()
                api_health = response.json()
                health["checks"]["api"] = {
                    "status": "healthy",
                    "response": api_health
                }
            except Exception as e:
                health["checks"]["api"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

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

    # ============================================================================
    # AUTONOMOUS AGENT FEATURES (v1.3.0)
    # ============================================================================

    def create_working_memory_session(self, session_id: str) -> Dict[str, Any]:
        """
        Create a working memory session for autonomous agent operation.

        Working memory provides short-term, task-specific memory that persists
        only for the duration of a task or session. Ideal for multi-step reasoning.

        Args:
            session_id: Unique identifier for the working memory session

        Returns:
            Session info including session_id, created_at, and ttl

        Raises:
            ValueError: If autonomous features not enabled or invalid session_id

        Example:
            session = memory.create_working_memory_session("task-123")
            # Perform multi-step reasoning with working memory
            memory.end_working_memory_session("task-123")
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled. Set enable_autonomous=True")

        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id must be a non-empty string")

        # Check for existing session
        if session_id in self._working_memory_sessions:
            if self.enable_logging:
                logger.warning(f"Working memory session '{session_id}' already exists, reusing")
            return self._working_memory_sessions[session_id]

        session_data = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl": self._autonomous_config["working_memory_ttl"],
            "items": [],
            "context": {},
            "active": True
        }

        with self._lock:
            self._working_memory_sessions[session_id] = session_data

        if self.enable_logging:
            logger.info(f"Created working memory session: {session_id}")

        return session_data

    def add_to_working_memory(self, session_id: str, item: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add an item to working memory session.

        Args:
            session_id: Working memory session ID
            item: Content to add to working memory
            metadata: Optional metadata for the item

        Raises:
            ValueError: If session doesn't exist
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled")

        if session_id not in self._working_memory_sessions:
            raise ValueError(f"Working memory session '{session_id}' not found")

        with self._lock:
            self._working_memory_sessions[session_id]["items"].append({
                "content": item,
                "metadata": metadata or {},
                "added_at": datetime.now(timezone.utc).isoformat()
            })

    def get_working_memory(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all items from a working memory session.

        Args:
            session_id: Working memory session ID

        Returns:
            List of working memory items

        Raises:
            ValueError: If session doesn't exist
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled")

        if session_id not in self._working_memory_sessions:
            raise ValueError(f"Working memory session '{session_id}' not found")

        return self._working_memory_sessions[session_id]["items"].copy()

    def end_working_memory_session(self, session_id: str, persist: bool = False) -> None:
        """
        End a working memory session.

        Args:
            session_id: Working memory session ID
            persist: If True, persist working memory items to long-term memory

        Raises:
            ValueError: If session doesn't exist
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled")

        if session_id not in self._working_memory_sessions:
            raise ValueError(f"Working memory session '{session_id}' not found")

        session = self._working_memory_sessions[session_id]

        # Optionally persist to long-term memory
        if persist and session["items"]:
            summary = f"Working memory session {session_id}: " + " | ".join(
                item["content"] for item in session["items"][:10]  # Limit to first 10 items
            )
            try:
                self.learn(summary, source="working_memory")
            except Exception as e:
                if self.enable_logging:
                    logger.error(f"Failed to persist working memory: {e}")

        with self._lock:
            del self._working_memory_sessions[session_id]

        if self.enable_logging:
            logger.info(f"Ended working memory session: {session_id}")

    def track_goal(self, goal_id: str, steps: List[str]) -> Dict[str, Any]:
        """
        Track a goal with multiple steps for autonomous agent operation.

        Goal tracking enables agents to maintain awareness of their objectives
        and track progress through multi-step tasks.

        Args:
            goal_id: Unique identifier for the goal
            steps: List of steps to complete the goal

        Returns:
            Goal tracking info including goal_id, steps, and progress

        Raises:
            ValueError: If autonomous features not enabled

        Example:
            goal = memory.track_goal("search-task", [
                "Gather requirements",
                "Search documents",
                "Synthesize results",
                "Generate response"
            ])
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled. Set enable_autonomous=True")

        if not self._autonomous_config["goal_tracking_enabled"]:
            raise ValueError("Goal tracking is disabled in autonomous_features config")

        if not goal_id or not isinstance(goal_id, str):
            raise ValueError("goal_id must be a non-empty string")

        if not steps or not isinstance(steps, list):
            raise ValueError("steps must be a non-empty list")

        goal_data = {
            "goal_id": goal_id,
            "steps": [{"step": step, "status": "pending", "completed_at": None} for step in steps],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_step": 0,
            "total_steps": len(steps),
            "status": "in_progress",
            "progress": 0.0
        }

        with self._lock:
            self._active_goals[goal_id] = goal_data

        if self.enable_logging:
            logger.info(f"Started tracking goal '{goal_id}' with {len(steps)} steps")

        return goal_data

    def complete_goal_step(self, goal_id: str, step_index: int = None) -> Dict[str, Any]:
        """
        Mark a goal step as complete.

        Args:
            goal_id: Goal identifier
            step_index: Index of step to complete (defaults to current step)

        Returns:
            Updated goal data

        Raises:
            ValueError: If goal doesn't exist
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled")

        if goal_id not in self._active_goals:
            raise ValueError(f"Goal '{goal_id}' not found")

        with self._lock:
            goal = self._active_goals[goal_id]
            idx = step_index if step_index is not None else goal["current_step"]

            if idx >= len(goal["steps"]):
                raise ValueError(f"Step index {idx} out of range")

            goal["steps"][idx]["status"] = "completed"
            goal["steps"][idx]["completed_at"] = datetime.now(timezone.utc).isoformat()

            # Update current step to next pending step
            completed_count = sum(1 for s in goal["steps"] if s["status"] == "completed")
            goal["progress"] = completed_count / goal["total_steps"]

            if completed_count == goal["total_steps"]:
                goal["status"] = "completed"
                goal["completed_at"] = datetime.now(timezone.utc).isoformat()
            else:
                # Find next pending step
                for i, step in enumerate(goal["steps"]):
                    if step["status"] == "pending":
                        goal["current_step"] = i
                        break

        if self.enable_logging:
            logger.info(f"Goal '{goal_id}' progress: {goal['progress']*100:.0f}%")

        return self._active_goals[goal_id].copy()

    def get_goal_status(self, goal_id: str) -> Dict[str, Any]:
        """
        Get the current status of a tracked goal.

        Args:
            goal_id: Goal identifier

        Returns:
            Goal status including progress and current step

        Raises:
            ValueError: If goal doesn't exist
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled")

        if goal_id not in self._active_goals:
            raise ValueError(f"Goal '{goal_id}' not found")

        return self._active_goals[goal_id].copy()

    def assess_quality(self, response: str, confidence: float) -> Dict[str, Any]:
        """
        Assess the quality of a response for metacognitive awareness.

        Quality assessment enables agents to evaluate their own outputs,
        supporting self-improvement and uncertainty awareness.

        Args:
            response: The response text to assess
            confidence: Agent's confidence in the response (0.0 to 1.0)

        Returns:
            Quality assessment including confidence, quality_score, and recommendations

        Raises:
            ValueError: If autonomous features not enabled

        Example:
            assessment = memory.assess_quality(
                response="The capital of France is Paris.",
                confidence=0.95
            )
            if assessment["quality_score"] < 0.7:
                # Consider regenerating response
                pass
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled. Set enable_autonomous=True")

        if not self._autonomous_config["metacognition_enabled"]:
            raise ValueError("Metacognition is disabled in autonomous_features config")

        if not isinstance(response, str):
            raise ValueError("response must be a string")

        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be a number between 0 and 1")

        # Perform quality assessment
        response_length = len(response)
        threshold = self._autonomous_config["confidence_threshold"]

        # Quality heuristics
        quality_factors = []

        # Length factor (very short responses may be incomplete)
        if response_length < 10:
            quality_factors.append(("length", 0.3, "Response is very short"))
        elif response_length < 50:
            quality_factors.append(("length", 0.6, "Response is brief"))
        else:
            quality_factors.append(("length", 1.0, "Response has adequate length"))

        # Confidence factor
        if confidence >= threshold:
            quality_factors.append(("confidence", 1.0, "Confidence meets threshold"))
        else:
            quality_factors.append(("confidence", confidence / threshold, f"Confidence below threshold ({threshold})"))

        # Calculate overall quality score
        quality_score = sum(f[1] for f in quality_factors) / len(quality_factors)

        assessment = {
            "response_length": response_length,
            "confidence": confidence,
            "confidence_threshold": threshold,
            "quality_score": quality_score,
            "meets_quality_bar": quality_score >= threshold,
            "factors": [{"factor": f[0], "score": f[1], "reason": f[2]} for f in quality_factors],
            "recommendations": []
        }

        # Add recommendations
        if confidence < threshold:
            assessment["recommendations"].append("Consider gathering more context before responding")
        if response_length < 50:
            assessment["recommendations"].append("Consider providing more detail in the response")
        if quality_score < threshold:
            assessment["recommendations"].append("Response may benefit from revision")

        if self.enable_logging:
            logger.debug(f"Quality assessment: score={quality_score:.2f}, confidence={confidence:.2f}")

        return assessment

    def quantify_uncertainty(
        self,
        response: str,
        confidence: float,
        evidence: List[str] = None
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty in a response for metacognitive awareness.

        Uncertainty quantification enables agents to express appropriate
        epistemic humility and identify areas needing more information.

        Args:
            response: The response text to analyze
            confidence: Agent's confidence in the response (0.0 to 1.0)
            evidence: List of supporting evidence or sources

        Returns:
            Uncertainty analysis including uncertainty_score, evidence_strength, and gaps

        Raises:
            ValueError: If autonomous features not enabled

        Example:
            uncertainty = memory.quantify_uncertainty(
                response="The project deadline is likely next Friday.",
                confidence=0.7,
                evidence=["Email from manager mentioning Friday", "Calendar invite"]
            )
            if uncertainty["uncertainty_score"] > 0.3:
                print(f"Gaps: {uncertainty['knowledge_gaps']}")
        """
        if not self.enable_autonomous:
            raise ValueError("Autonomous features not enabled. Set enable_autonomous=True")

        if not self._autonomous_config["metacognition_enabled"]:
            raise ValueError("Metacognition is disabled in autonomous_features config")

        if not isinstance(response, str):
            raise ValueError("response must be a string")

        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be a number between 0 and 1")

        evidence = evidence or []

        # Calculate uncertainty score (inverse of confidence adjusted by evidence)
        base_uncertainty = 1.0 - confidence

        # Evidence strength calculation
        evidence_count = len(evidence)
        if evidence_count == 0:
            evidence_strength = 0.0
            evidence_adjustment = 0.2  # Increase uncertainty without evidence
        elif evidence_count == 1:
            evidence_strength = 0.5
            evidence_adjustment = 0.0
        elif evidence_count <= 3:
            evidence_strength = 0.75
            evidence_adjustment = -0.1  # Decrease uncertainty with moderate evidence
        else:
            evidence_strength = 1.0
            evidence_adjustment = -0.15  # Decrease uncertainty with strong evidence

        uncertainty_score = max(0.0, min(1.0, base_uncertainty + evidence_adjustment))

        # Detect uncertainty indicators in response
        uncertainty_indicators = [
            "probably", "likely", "might", "may", "could",
            "possibly", "perhaps", "uncertain", "unclear", "unsure",
            "approximately", "around", "about", "roughly"
        ]
        response_lower = response.lower()
        detected_indicators = [ind for ind in uncertainty_indicators if ind in response_lower]

        # Identify knowledge gaps
        knowledge_gaps = []
        if confidence < 0.5:
            knowledge_gaps.append("Low confidence suggests significant knowledge gaps")
        if evidence_count == 0:
            knowledge_gaps.append("No supporting evidence provided")
        if len(detected_indicators) > 2:
            knowledge_gaps.append("Response contains multiple uncertainty markers")

        result = {
            "uncertainty_score": uncertainty_score,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "evidence_strength": evidence_strength,
            "uncertainty_indicators": detected_indicators,
            "knowledge_gaps": knowledge_gaps,
            "should_seek_clarification": uncertainty_score > 0.5,
            "evidence_provided": evidence
        }

        if self.enable_logging:
            logger.debug(f"Uncertainty quantification: score={uncertainty_score:.2f}, evidence={evidence_count}")

        return result

    @contextmanager
    def with_working_memory(self, session_id: str) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for working memory sessions.

        Automatically creates and cleans up working memory sessions,
        ensuring proper resource management.

        Args:
            session_id: Unique identifier for the working memory session

        Yields:
            Working memory session data

        Example:
            with memory.with_working_memory("task-123") as session:
                memory.add_to_working_memory("task-123", "Step 1 result")
                memory.add_to_working_memory("task-123", "Step 2 result")
                # Working memory automatically cleaned up after block
        """
        session = self.create_working_memory_session(session_id)
        try:
            yield session
        finally:
            try:
                self.end_working_memory_session(session_id, persist=False)
            except ValueError:
                pass  # Session may already be ended

    def with_goal_tracking(self, goal_id: str, steps: List[str]):
        """
        Decorator for goal tracking across function execution.

        Wraps a function to automatically track goal progress,
        marking steps complete as the function progresses.

        Args:
            goal_id: Unique identifier for the goal
            steps: List of steps to track

        Returns:
            Decorator function

        Example:
            @memory.with_goal_tracking("analysis-task", ["fetch", "analyze", "report"])
            def analyze_data():
                # Step 1: fetch
                data = fetch_data()
                # Function can call complete_goal_step to mark progress
                return data
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Start goal tracking
                self.track_goal(goal_id, steps)

                try:
                    result = func(*args, **kwargs)

                    # Auto-complete all remaining steps on success
                    goal = self.get_goal_status(goal_id)
                    for i, step in enumerate(goal["steps"]):
                        if step["status"] == "pending":
                            self.complete_goal_step(goal_id, i)

                    return result

                except Exception as e:
                    # Mark goal as failed
                    with self._lock:
                        if goal_id in self._active_goals:
                            self._active_goals[goal_id]["status"] = "failed"
                            self._active_goals[goal_id]["error"] = str(e)
                    raise

            return wrapper
        return decorator

    def get_autonomous_status(self) -> Dict[str, Any]:
        """
        Get status of all autonomous agent features.

        Returns:
            Status of working memory sessions, active goals, and configuration
        """
        if not self.enable_autonomous:
            return {"enabled": False}

        return {
            "enabled": True,
            "config": self._autonomous_config.copy(),
            "working_memory_sessions": len(self._working_memory_sessions),
            "active_sessions": list(self._working_memory_sessions.keys()),
            "active_goals": len(self._active_goals),
            "goals": {
                goal_id: {
                    "status": goal["status"],
                    "progress": goal["progress"],
                    "current_step": goal["current_step"]
                }
                for goal_id, goal in self._active_goals.items()
            }
        }
