"""RecallBricks Usage API

Usage tracking, billing information, and consumption monitoring.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging

from .client import RecallBricksClient


logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class UsageStats:
    """Current month usage statistics."""
    operations_used: int = 0
    operations_limit: int = 0
    operations_remaining: int = 0
    percent_used: float = 0.0
    cost_usd: str = "0.00"
    in_overage: bool = False
    status: str = "healthy"  # healthy, warning, overage, blocked
    message: str = ""
    plan: str = "free"  # free, pro, team, enterprise
    month: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageStats":
        return cls(
            operations_used=data.get("operations_used", 0),
            operations_limit=data.get("operations_limit", 0),
            operations_remaining=data.get("operations_remaining", 0),
            percent_used=data.get("percent_used", 0.0),
            cost_usd=data.get("cost_usd", "0.00"),
            in_overage=data.get("in_overage", False),
            status=data.get("status", "healthy"),
            message=data.get("message", ""),
            plan=data.get("plan", "free"),
            month=data.get("month", ""),
        )


@dataclass
class UsageHistoryEntry:
    """Historical usage entry for a month."""
    month: str
    operations_count: int = 0
    cost_cents: int = 0
    plan: str = "free"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageHistoryEntry":
        return cls(
            month=data.get("month", ""),
            operations_count=data.get("operations_count", 0),
            cost_cents=data.get("cost_cents", 0),
            plan=data.get("plan", "free"),
        )


@dataclass
class UsageBreakdown:
    """Usage breakdown by event type."""
    event_type: str
    count: int = 0
    percentage: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageBreakdown":
        return cls(
            event_type=data.get("event_type", data.get("type", "")),
            count=data.get("count", 0),
            percentage=data.get("percentage", 0.0),
        )


# ============================================================================
# USAGE API
# ============================================================================

class UsageAPI:
    """
    Usage tracking and billing information.

    Features:
    - Current month usage statistics
    - Historical usage data
    - Usage breakdown by operation type
    - Plan limits and overage status

    Plans:
    - free: 1,000 operations/month
    - pro: 10,000 operations/month
    - team: 100,000 operations/month
    - enterprise: Unlimited

    Example:
        from recallbricks_langchain import RecallBricksClient, UsageAPI

        client = RecallBricksClient(api_key="your-key")
        usage = UsageAPI(client)

        # Check current usage
        stats = usage.get_current()
        print(f"Used: {stats.operations_used}/{stats.operations_limit}")
        print(f"Cost: ${stats.cost_usd}")

        if stats.status == "warning":
            print("Warning: Approaching usage limit!")

        # Get historical usage
        history = usage.get_history()
        for entry in history:
            print(f"{entry.month}: {entry.operations_count} operations")
    """

    def __init__(self, client: RecallBricksClient):
        self.client = client

    def get_current(self, user_id: str = None) -> UsageStats:
        """
        Get current month usage statistics.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            UsageStats with current consumption

        Status Values:
        - healthy: Under 80% of limit
        - warning: 80-100% of limit
        - overage: Over limit (pay-as-you-go)
        - blocked: Over limit (no overage allowed)
        """
        result = self.client.get("/usage/current", user_id=user_id)
        return UsageStats.from_dict(result)

    def get_history(
        self,
        months: int = 6,
        user_id: str = None,
    ) -> List[UsageHistoryEntry]:
        """
        Get usage history for past months.

        Args:
            months: Number of months to retrieve (default: 6)
            user_id: User ID (required for service token auth)

        Returns:
            List of UsageHistoryEntry objects
        """
        params = {"months": months}
        result = self.client.get("/usage/history", params=params, user_id=user_id)
        history = result.get("history", result.get("items", []))
        return [UsageHistoryEntry.from_dict(h) for h in history]

    def get_breakdown(
        self,
        month: str = None,
        user_id: str = None,
    ) -> List[UsageBreakdown]:
        """
        Get usage breakdown by event type.

        Args:
            month: Month to analyze (YYYY-MM format, default: current)
            user_id: User ID (required for service token auth)

        Returns:
            List of UsageBreakdown by operation type

        Event Types:
        - memory_create: Memory creation
        - memory_learn: Smart learn with extraction
        - memory_search: Semantic search
        - memory_recall: Organized recall
        - context_get: Context retrieval
        - relationship_detect: Relationship detection
        """
        params = {}
        if month:
            params["month"] = month

        result = self.client.get("/usage/breakdown", params=params, user_id=user_id)
        breakdown = result.get("breakdown", result.get("items", []))
        return [UsageBreakdown.from_dict(b) for b in breakdown]

    def check_limits(self, user_id: str = None) -> Dict[str, Any]:
        """
        Quick check of usage limits.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Dictionary with limit status

        Example:
            limits = usage.check_limits()
            if limits['can_proceed']:
                # Safe to make requests
                pass
            else:
                print(f"Blocked: {limits['reason']}")
        """
        stats = self.get_current(user_id)

        return {
            "can_proceed": stats.status not in ["blocked"],
            "status": stats.status,
            "percent_used": stats.percent_used,
            "remaining": stats.operations_remaining,
            "reason": stats.message if stats.status in ["warning", "overage", "blocked"] else None,
        }
