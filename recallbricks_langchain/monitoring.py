"""RecallBricks Monitoring API

Health checks, metrics, SLA tracking, and audit logs.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .client import RecallBricksClient


logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HealthCheck:
    """Health check result."""
    status: str = "healthy"  # healthy, degraded, unhealthy
    timestamp: str = ""
    version: Optional[str] = None
    components: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthCheck":
        return cls(
            status=data.get("status", "healthy"),
            timestamp=data.get("timestamp", ""),
            version=data.get("version"),
            components=data.get("components", data.get("checks", {})),
        )


@dataclass
class SLAMetrics:
    """SLA metrics for a time period."""
    period: str
    uptime_percent: float = 100.0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    error_rate: float = 0.0
    requests_per_hour: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SLAMetrics":
        return cls(
            period=data.get("period", ""),
            uptime_percent=data.get("uptime_percent", data.get("uptime", 100.0)),
            avg_response_time_ms=data.get("avg_response_time_ms", data.get("avg_response_time", 0.0)),
            p95_response_time_ms=data.get("p95_response_time_ms", data.get("p95", 0.0)),
            p99_response_time_ms=data.get("p99_response_time_ms", data.get("p99", 0.0)),
            error_rate=data.get("error_rate", 0.0),
            requests_per_hour=data.get("requests_per_hour", data.get("rpm", 0.0) * 60),
        )


@dataclass
class AuditLogEntry:
    """Audit log entry."""
    id: str
    user_id: str
    action_type: str
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    timestamp: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogEntry":
        return cls(
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
            action_type=data.get("action_type", data.get("action", "")),
            resource_id=data.get("resource_id"),
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class ComponentHealth:
    """Individual component health status."""
    name: str
    status: str = "healthy"
    latency_ms: float = 0.0
    message: Optional[str] = None
    last_check: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentHealth":
        return cls(
            name=data.get("name", data.get("component", "")),
            status=data.get("status", "healthy"),
            latency_ms=data.get("latency_ms", data.get("latency", 0.0)),
            message=data.get("message"),
            last_check=data.get("last_check", data.get("timestamp", "")),
        )


@dataclass
class DashboardMetrics:
    """Dashboard metrics overview."""
    total_memories: int = 0
    memories_by_tier: Dict[str, int] = field(default_factory=dict)
    total_relationships: int = 0
    avg_importance: float = 0.0
    storage_used_mb: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DashboardMetrics":
        return cls(
            total_memories=data.get("total_memories", data.get("total", 0)),
            memories_by_tier=data.get("memories_by_tier", data.get("by_tier", {})),
            total_relationships=data.get("total_relationships", 0),
            avg_importance=data.get("avg_importance", 0.0),
            storage_used_mb=data.get("storage_used_mb", data.get("storage_mb", 0.0)),
        )


@dataclass
class SystemInsight:
    """System observer insight/anomaly."""
    id: str
    insight_type: str
    severity: str  # info, warning, critical
    description: str
    resolved: bool = False
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemInsight":
        return cls(
            id=data.get("id", ""),
            insight_type=data.get("insight_type", data.get("type", "")),
            severity=data.get("severity", "info"),
            description=data.get("description", ""),
            resolved=data.get("resolved", False),
            created_at=data.get("created_at", ""),
        )


# ============================================================================
# MONITORING API
# ============================================================================

class MonitoringAPI:
    """
    Health checks, metrics, SLA tracking, and system observability.

    Features:
    - Health checks (liveness, readiness, components)
    - Prometheus metrics export
    - SLA tracking and reporting
    - Audit logs
    - System insights and anomaly detection

    Example:
        from recallbricks_langchain import RecallBricksClient, MonitoringAPI

        client = RecallBricksClient(api_key="your-key")
        monitoring = MonitoringAPI(client)

        # Check system health
        health = monitoring.health()
        print(f"Status: {health.status}")
        for name, component in health.components.items():
            print(f"  {name}: {component.get('status')}")

        # Get SLA metrics
        sla = monitoring.get_sla(period="7d")
        print(f"Uptime: {sla.uptime_percent}%")
        print(f"P99 Latency: {sla.p99_response_time_ms}ms")

        # Get Prometheus metrics
        prometheus = monitoring.get_prometheus_metrics()
        # Export to monitoring system
    """

    def __init__(self, client: RecallBricksClient):
        self.client = client

    # =========================================================================
    # Health Checks
    # =========================================================================

    def health(self, user_id: str = None) -> HealthCheck:
        """
        Get comprehensive health status.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            HealthCheck with component statuses
        """
        result = self.client.get("/monitoring/health", user_id=user_id)
        return HealthCheck.from_dict(result)

    def health_simple(self) -> Dict[str, str]:
        """
        Get simple health status for load balancers.

        No authentication required.

        Returns:
            Simple status dictionary
        """
        return self.client.get("/monitoring/health/simple")

    def ready(self) -> bool:
        """
        Check readiness (database connectivity, circuit breaker).

        No authentication required.

        Returns:
            True if ready to serve requests
        """
        try:
            result = self.client.get("/monitoring/ready")
            return result.get("ready", result.get("status") == "ready")
        except Exception:
            return False

    def live(self) -> bool:
        """
        Check liveness (application running).

        No authentication required.

        Returns:
            True if application is alive
        """
        try:
            result = self.client.get("/monitoring/live")
            return result.get("live", result.get("status") == "ok")
        except Exception:
            return False

    def get_component_health(
        self,
        component: str,
        user_id: str = None,
    ) -> ComponentHealth:
        """
        Get health status for a specific component.

        Args:
            component: Component name (database, cache, embeddings, etc.)
            user_id: User ID (required for service token auth)

        Returns:
            ComponentHealth for the specified component
        """
        result = self.client.get(f"/monitoring/components/{component}", user_id=user_id)
        return ComponentHealth.from_dict(result)

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_prometheus_metrics(self, user_id: str = None) -> str:
        """
        Get metrics in Prometheus format.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Prometheus-formatted metrics string

        Metrics Include:
        - recallbricks_http_requests_total
        - recallbricks_http_request_duration_seconds
        - recallbricks_memory_operations_total
        - recallbricks_circuit_breaker_state
        """
        result = self.client.get("/monitoring/metrics", user_id=user_id)
        if isinstance(result, str):
            return result
        return result.get("metrics", "")

    def get_metrics_json(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get metrics in JSON format.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Dictionary with all metrics
        """
        return self.client.get("/monitoring/metrics/json", user_id=user_id)

    # =========================================================================
    # SLA
    # =========================================================================

    def get_sla(
        self,
        period: str = "24h",
        user_id: str = None,
    ) -> SLAMetrics:
        """
        Get SLA metrics for a time period.

        Args:
            period: Time period ('1h', '24h', '7d', '30d')
            user_id: User ID (required for service token auth)

        Returns:
            SLAMetrics with uptime, latency, and error rates
        """
        params = {"period": period}
        result = self.client.get("/monitoring/sla", params=params, user_id=user_id)
        return SLAMetrics.from_dict(result)

    # =========================================================================
    # Audit Logs
    # =========================================================================

    def get_audit_logs(
        self,
        start_date: str = None,
        end_date: str = None,
        action_type: str = None,
        limit: int = 100,
        offset: int = 0,
        user_id: str = None,
    ) -> List[AuditLogEntry]:
        """
        Get audit logs.

        Args:
            start_date: Start date (ISO 8601)
            end_date: End date (ISO 8601)
            action_type: Filter by action type
            limit: Maximum results
            offset: Pagination offset
            user_id: User ID (required for service token auth)

        Returns:
            List of AuditLogEntry objects
        """
        params = {"limit": limit, "offset": offset}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if action_type:
            params["action_type"] = action_type

        result = self.client.get("/monitoring/audit/logs", params=params, user_id=user_id)
        logs = result.get("logs", result.get("items", []))
        return [AuditLogEntry.from_dict(l) for l in logs]

    def get_audit_stats(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Dictionary with audit statistics
        """
        return self.client.get("/monitoring/audit/stats", user_id=user_id)

    # =========================================================================
    # Dashboard
    # =========================================================================

    def get_dashboard_metrics(
        self,
        range: str = "24h",
        user_id: str = None,
    ) -> DashboardMetrics:
        """
        Get dashboard metrics overview.

        Args:
            range: Time range ('1h', '24h', '7d', '30d')
            user_id: User ID (required for service token auth)

        Returns:
            DashboardMetrics with tier breakdown
        """
        params = {"range": range}
        result = self.client.get("/dashboard/metrics", params=params, user_id=user_id)
        return DashboardMetrics.from_dict(result)

    def get_dashboard_stats(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get dashboard statistics.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Dictionary with memory stats
        """
        return self.client.get("/dashboard/stats", user_id=user_id)

    def get_upgrade_status(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get tier upgrade status.

        Shows how many memories are pending upgrade from Tier 1 to Tier 2.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Dictionary with upgrade queue status
        """
        return self.client.get("/dashboard/upgrade-status", user_id=user_id)

    def get_costs(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get cost breakdown.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Dictionary with cost breakdown by operation
        """
        return self.client.get("/dashboard/costs", user_id=user_id)

    # =========================================================================
    # System Insights
    # =========================================================================

    def get_telemetry(
        self,
        timeframe: str = "1h",
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Get system telemetry.

        Args:
            timeframe: Time frame ('1h', '24h', '7d')
            user_id: User ID (required for service token auth)

        Returns:
            Dictionary with telemetry data
        """
        params = {"timeframe": timeframe}
        return self.client.get("/system/telemetry", params=params, user_id=user_id)

    def get_insights(self, user_id: str = None) -> List[SystemInsight]:
        """
        Get system observer insights.

        Returns anomalies and patterns detected by self-observation.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            List of SystemInsight objects
        """
        result = self.client.get("/system/insights", user_id=user_id)
        insights = result.get("insights", result.get("items", []))
        return [SystemInsight.from_dict(i) for i in insights]

    def resolve_insight(
        self,
        insight_id: str,
        user_id: str = None,
    ) -> SystemInsight:
        """
        Mark an insight as resolved.

        Args:
            insight_id: Insight UUID
            user_id: User ID (required for service token auth)

        Returns:
            Updated SystemInsight
        """
        result = self.client.post(f"/system/insights/{insight_id}/resolve", {}, user_id=user_id)
        return SystemInsight.from_dict(result)

    def discover_patterns(self, user_id: str = None) -> Dict[str, Any]:
        """
        Trigger anomaly pattern discovery.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Discovered patterns
        """
        return self.client.post("/system/patterns/discover", {}, user_id=user_id)
