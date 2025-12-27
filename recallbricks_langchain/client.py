"""RecallBricks Base Client

Enterprise-grade HTTP client with connection pooling, rate limiting,
circuit breaker, retry logic, and distributed tracing.
"""

from typing import Any, Dict, Optional, Callable, Union
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
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
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
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class CircuitBreakerError(RecallBricksError):
    """Raised when circuit breaker is open."""
    pass


class APIError(RecallBricksError):
    """Raised when API returns an error."""
    def __init__(self, message: str, status_code: int = None, error_code: str = None, **kwargs):
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message, **kwargs)


class TimeoutError(RecallBricksError):
    """Raised when request times out."""
    pass


class DeduplicationError(RecallBricksError):
    """Raised when duplicate request is detected."""
    pass


class AuthenticationError(RecallBricksError):
    """Raised when authentication fails."""
    pass


class NotFoundError(RecallBricksError):
    """Raised when resource is not found."""
    pass


class ServiceUnavailableError(RecallBricksError):
    """Raised when service is unavailable (circuit breaker open on server)."""
    pass


# ============================================================================
# RATE LIMIT INFO
# ============================================================================

@dataclass
class RateLimitInfo:
    """Rate limit information from API response headers."""
    limit: int = 0
    remaining: int = 0
    reset: Optional[datetime] = None
    retry_after: Optional[int] = None


# ============================================================================
# SHARED SESSION - Connection pooling
# ============================================================================

_session = None
_session_lock = threading.Lock()


def get_session() -> requests.Session:
    """
    Get shared requests session with connection pooling.
    Reuses TCP connections across requests for performance.
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


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for handling API failures gracefully.
    Prevents cascading failures by temporarily blocking requests after failures.

    States:
    - closed: Normal operation, requests go through
    - open: Failures exceeded threshold, requests blocked
    - half_open: Testing if service recovered
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
        self.state = "closed"
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if datetime.now(timezone.utc) - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                    self.state = "half_open"
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN - service temporarily unavailable")
            state_before_call = self.state

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


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API request throttling.
    Prevents API abuse and respects server rate limits.
    """

    def __init__(self, rate: int = 100, per: int = 60):
        self.rate = rate
        self.per = per
        self.allowance = float(rate)
        self.last_check = time.time()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Check if request is allowed under rate limit."""
        with self._lock:
            current = time.time()
            elapsed = current - self.last_check
            self.last_check = current
            self.allowance += elapsed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate
            if self.allowance < 1.0:
                return False
            self.allowance -= 1.0
            return True

    def wait_if_needed(self):
        """Wait if rate limit exceeded, then allow request."""
        while not self.allow():
            time.sleep(0.1)


# ============================================================================
# REQUEST DEDUPLICATOR
# ============================================================================

class RequestDeduplicator:
    """Request deduplication using content hashing with sliding window."""

    def __init__(self, window_size: int = 1000, window_seconds: int = 60):
        self.window_size = window_size
        self.window_seconds = window_seconds
        self.recent_requests = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def _hash_request(self, data: Dict[str, Any]) -> str:
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def is_duplicate(self, data: Dict[str, Any]) -> bool:
        request_hash = self._hash_request(data)
        current_time = time.time()
        with self._lock:
            cutoff_time = current_time - self.window_seconds
            while self.recent_requests and self.recent_requests[0][1] < cutoff_time:
                self.recent_requests.popleft()
            for stored_hash, timestamp in self.recent_requests:
                if stored_hash == request_hash:
                    return True
            self.recent_requests.append((request_hash, current_time))
            return False


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """Prometheus-compatible metrics collector with thread-safe aggregation."""

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
        self.response_times = deque(maxlen=1000)
        self._lock = threading.Lock()

    def increment(self, metric: str, value: int = 1):
        with self._lock:
            if metric in self.metrics:
                self.metrics[metric] += value

    def record_response_time(self, duration: float):
        with self._lock:
            self.metrics["response_time_sum"] += duration
            self.metrics["response_time_count"] += 1
            self.response_times.append(duration)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            metrics = self.metrics.copy()
            if self.response_times:
                sorted_times = sorted(self.response_times)
                metrics["response_time_p50"] = sorted_times[len(sorted_times) // 2]
                metrics["response_time_p95"] = sorted_times[int(len(sorted_times) * 0.95)]
                metrics["response_time_p99"] = sorted_times[int(len(sorted_times) * 0.99)]
            if metrics["response_time_count"] > 0:
                metrics["response_time_avg"] = metrics["response_time_sum"] / metrics["response_time_count"]
            if metrics["requests_total"] > 0:
                metrics["success_rate"] = metrics["requests_success"] / metrics["requests_total"]
            return metrics

    def export_prometheus(self) -> str:
        metrics = self.get_metrics()
        lines = []
        for key, value in metrics.items():
            metric_name = f"recallbricks_{key}"
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")
        return "\n".join(lines)


# ============================================================================
# BASE CLIENT
# ============================================================================

class RecallBricksClient:
    """
    Enterprise-grade base client for RecallBricks API.

    Features:
    - Connection pooling for performance
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Rate limiting (client-side)
    - Server rate limit header handling
    - Request deduplication
    - Distributed tracing with request IDs
    - Prometheus metrics collection
    - Graceful shutdown

    Authentication:
    - API Key (X-API-Key header) - for single-tenant apps
    - Service Token (X-Service-Token header) - for multi-tenant apps
    """

    def __init__(
        self,
        api_key: str = None,
        service_token: str = None,
        api_url: str = "https://api.recallbricks.com/api/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
        rate_limit: int = 100,
        rate_limit_period: int = 60,
        enable_deduplication: bool = True,
        enable_metrics: bool = True,
        enable_distributed_tracing: bool = True,
        enable_logging: bool = True,
    ):
        """
        Initialize RecallBricks client.

        Args:
            api_key: API key for authentication (single-tenant)
            service_token: Service token for authentication (multi-tenant)
            api_url: Base URL for RecallBricks API
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay for exponential backoff
            timeout: Request timeout in seconds
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Seconds before attempting recovery
            rate_limit: Maximum requests per period
            rate_limit_period: Period in seconds
            enable_deduplication: Enable request deduplication
            enable_metrics: Enable Prometheus metrics collection
            enable_distributed_tracing: Enable request ID tracking
            enable_logging: Enable detailed logging
        """
        # Validate authentication
        self.api_key = api_key or os.getenv("RECALLBRICKS_API_KEY")
        self.service_token = service_token or os.getenv("RECALLBRICKS_SERVICE_TOKEN")

        if not self.api_key and not self.service_token:
            raise AuthenticationError(
                "Either api_key or service_token must be provided, "
                "or set RECALLBRICKS_API_KEY or RECALLBRICKS_SERVICE_TOKEN environment variable"
            )

        # Validate HTTPS
        if not api_url.startswith('https://'):
            raise ValidationError(f"api_url must use HTTPS for security. Got: {api_url}")

        self.api_url = api_url.rstrip('/')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.enable_logging = enable_logging
        self.enable_distributed_tracing = enable_distributed_tracing

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_timeout
        )

        # Rate limiter
        self.rate_limiter = RateLimiter(rate=rate_limit, per=rate_limit_period)

        # Server rate limit info
        self.server_rate_limit = RateLimitInfo()

        # Deduplication
        self.enable_deduplication = enable_deduplication
        if enable_deduplication:
            self.deduplicator = RequestDeduplicator()

        # Metrics
        self.enable_metrics = enable_metrics
        if enable_metrics:
            self.metrics = MetricsCollector()

        # Thread safety
        self._lock = threading.Lock()

        # Graceful shutdown
        self._shutdown = False
        atexit.register(self.shutdown)

        if enable_logging:
            logger.info(f"RecallBricksClient initialized: {api_url}")

    def _get_headers(self, user_id: str = None) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {"Content-Type": "application/json"}

        if self.service_token:
            headers["X-Service-Token"] = self.service_token
            if user_id:
                headers["X-User-ID"] = user_id
        elif self.api_key:
            headers["X-API-Key"] = self.api_key

        return headers

    def _parse_rate_limit_headers(self, response: requests.Response):
        """Parse rate limit information from response headers."""
        try:
            if "X-RateLimit-Limit" in response.headers:
                self.server_rate_limit.limit = int(response.headers["X-RateLimit-Limit"])
            if "X-RateLimit-Remaining" in response.headers:
                self.server_rate_limit.remaining = int(response.headers["X-RateLimit-Remaining"])
            if "X-RateLimit-Reset" in response.headers:
                self.server_rate_limit.reset = datetime.fromisoformat(
                    response.headers["X-RateLimit-Reset"].replace("Z", "+00:00")
                )
            if "Retry-After" in response.headers:
                self.server_rate_limit.retry_after = int(response.headers["Retry-After"])
        except (ValueError, TypeError):
            pass

    def _handle_error_response(self, response: requests.Response, request_id: str = None):
        """Handle error responses with proper exception types."""
        try:
            error_data = response.json().get("error", {})
            message = error_data.get("message", response.text)
            error_code = error_data.get("code", "UNKNOWN_ERROR")
        except Exception:
            message = response.text
            error_code = "UNKNOWN_ERROR"

        if response.status_code == 401:
            raise AuthenticationError(message, request_id=request_id, status_code=401)
        elif response.status_code == 403:
            raise AuthenticationError(f"Access forbidden: {message}", request_id=request_id, status_code=403)
        elif response.status_code == 404:
            raise NotFoundError(message, request_id=request_id, status_code=404)
        elif response.status_code == 429:
            retry_after = self.server_rate_limit.retry_after
            raise RateLimitError(
                f"Rate limit exceeded. Retry after {retry_after}s",
                retry_after=retry_after,
                request_id=request_id
            )
        elif response.status_code == 503:
            raise ServiceUnavailableError(message, request_id=request_id, status_code=503)
        else:
            raise APIError(
                message,
                status_code=response.status_code,
                error_code=error_code,
                request_id=request_id
            )

    def _execute_with_retry(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Execute HTTP request with retry logic and circuit breaker.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional request parameters (json, params, headers, user_id)

        Returns:
            JSON response data
        """
        request_id = str(uuid.uuid4()) if self.enable_distributed_tracing else None
        start_time = time.time()

        # Check shutdown
        if self._shutdown:
            raise RecallBricksError("Service is shutting down", request_id=request_id)

        # Track metrics
        if self.enable_metrics:
            self.metrics.increment("requests_total")

        # Check client rate limit
        if not self.rate_limiter.allow():
            if self.enable_metrics:
                self.metrics.increment("requests_rate_limited")
            raise RateLimitError("Client rate limit exceeded", request_id=request_id)

        # Check deduplication for POST requests
        if method == "POST" and self.enable_deduplication and "json" in kwargs:
            if hasattr(self, 'deduplicator') and self.deduplicator.is_duplicate(kwargs["json"]):
                if self.enable_metrics:
                    self.metrics.increment("requests_deduplicated")
                if self.enable_logging:
                    logger.warning(f"[{request_id}] Duplicate request detected, skipping")
                raise DeduplicationError("Duplicate request detected", request_id=request_id)

        # Build URL and headers
        url = f"{self.api_url}{endpoint}"
        user_id = kwargs.pop("user_id", None)
        headers = self._get_headers(user_id)
        headers.update(kwargs.pop("headers", {}))
        if request_id:
            headers["X-Request-ID"] = request_id

        retries = 0
        delay = self.retry_delay
        last_exception = None

        while retries <= self.max_retries:
            try:
                def make_request():
                    response = get_session().request(
                        method,
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        **kwargs
                    )
                    self._parse_rate_limit_headers(response)

                    if not response.ok:
                        self._handle_error_response(response, request_id)

                    # Handle empty responses
                    if response.status_code == 204 or not response.content:
                        return {}

                    return response.json()

                result = self.circuit_breaker.call(make_request)

                # Record success
                if self.enable_metrics:
                    self.metrics.increment("requests_success")
                    self.metrics.record_response_time(time.time() - start_time)

                return result

            except (RateLimitError, DeduplicationError, NotFoundError):
                raise

            except Exception as e:
                last_exception = e
                retries += 1

                if self.enable_metrics:
                    self.metrics.increment("retries_total")

                if retries > self.max_retries:
                    if self.enable_metrics:
                        self.metrics.increment("requests_failed")
                    if self.enable_logging:
                        logger.error(f"[{request_id}] Failed after {self.max_retries} retries: {e}")
                    raise

                # Exponential backoff with jitter
                import random
                jitter = random.uniform(0, 0.1 * delay)
                sleep_time = min(delay + jitter, 60.0)

                if self.enable_logging:
                    logger.warning(f"[{request_id}] Retry {retries}/{self.max_retries} after {sleep_time:.2f}s: {e}")

                time.sleep(sleep_time)
                delay *= 2

        raise last_exception

    def get(self, endpoint: str, params: Dict = None, **kwargs) -> Dict[str, Any]:
        """Execute GET request."""
        return self._execute_with_retry("GET", endpoint, params=params, **kwargs)

    def post(self, endpoint: str, data: Dict = None, **kwargs) -> Dict[str, Any]:
        """Execute POST request."""
        return self._execute_with_retry("POST", endpoint, json=data, **kwargs)

    def put(self, endpoint: str, data: Dict = None, **kwargs) -> Dict[str, Any]:
        """Execute PUT request."""
        return self._execute_with_retry("PUT", endpoint, json=data, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Execute DELETE request."""
        return self._execute_with_retry("DELETE", endpoint, **kwargs)

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check.

        Returns:
            Health status with detailed diagnostics
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }

        try:
            # Check API health
            try:
                api_health = self.get("/health")
                health["checks"]["api"] = {"status": "healthy", "response": api_health}
            except Exception as e:
                health["checks"]["api"] = {"status": "unhealthy", "error": str(e)}

            # Check circuit breaker
            cb_state = self.circuit_breaker.state
            health["checks"]["circuit_breaker"] = {
                "status": "healthy" if cb_state == "closed" else "degraded",
                "state": cb_state,
                "failure_count": self.circuit_breaker.failure_count
            }

            # Check rate limiter
            health["checks"]["rate_limiter"] = {
                "status": "healthy",
                "client_allowance": self.rate_limiter.allowance,
                "server_remaining": self.server_rate_limit.remaining
            }

            # Check metrics
            if self.enable_metrics:
                metrics = self.metrics.get_metrics()
                success_rate = metrics.get("success_rate", 1.0)
                health["checks"]["requests"] = {
                    "status": "healthy" if success_rate > 0.95 else "degraded",
                    "success_rate": success_rate,
                    "total": metrics.get("requests_total", 0)
                }

            # Overall status
            if any(c.get("status") == "unhealthy" for c in health["checks"].values()):
                health["status"] = "unhealthy"
            elif any(c.get("status") == "degraded" for c in health["checks"].values()):
                health["status"] = "degraded"

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if self.enable_metrics:
            return self.metrics.get_metrics()
        return {}

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if self.enable_metrics:
            return self.metrics.export_prometheus()
        return ""

    def shutdown(self):
        """Graceful shutdown."""
        if self._shutdown:
            return
        self._shutdown = True
        if self.enable_logging:
            logger.info("Shutting down RecallBricksClient...")
            if self.enable_metrics:
                logger.info(f"Final metrics: {json.dumps(self.metrics.get_metrics(), indent=2)}")
