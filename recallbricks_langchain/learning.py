"""RecallBricks Learning API

Metacognitive learning analysis, pattern detection, and self-improvement.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging

from .client import RecallBricksClient, ValidationError


logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RelationshipSuggestion:
    """Suggested relationship between memories."""
    memory_id: str
    related_memory_id: str
    suggested_type: str
    confidence: float
    reason: str
    co_access_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipSuggestion":
        return cls(
            memory_id=data.get("memory_id", ""),
            related_memory_id=data.get("related_memory_id", ""),
            suggested_type=data.get("suggested_type", "related_to"),
            confidence=data.get("confidence", 0.0),
            reason=data.get("reason", ""),
            co_access_count=data.get("co_access_count", 0),
        )


@dataclass
class LearningAnalysisResult:
    """Result from learning analysis."""
    timestamp: str
    clusters_detected: int = 0
    relationship_suggestions: List[RelationshipSuggestion] = field(default_factory=list)
    weight_adjustments: Dict[str, float] = field(default_factory=dict)
    stale_memory_count: int = 0
    processing_time_ms: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningAnalysisResult":
        suggestions = [
            RelationshipSuggestion.from_dict(s)
            for s in data.get("relationship_suggestions", [])
        ]
        return cls(
            timestamp=data.get("timestamp", ""),
            clusters_detected=data.get("clusters_detected", 0),
            relationship_suggestions=suggestions,
            weight_adjustments=data.get("weight_adjustments", {}),
            stale_memory_count=data.get("stale_memory_count", 0),
            processing_time_ms=data.get("processing_time_ms", 0),
        )


@dataclass
class MaintenanceSuggestion:
    """Suggestion for memory maintenance (archive, merge, quality)."""
    memory_id: str
    suggestion_type: str  # archive, merge, quality_check
    reason: str
    score: float = 0.0
    related_memory_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaintenanceSuggestion":
        return cls(
            memory_id=data.get("memory_id", ""),
            suggestion_type=data.get("suggestion_type", data.get("type", "")),
            reason=data.get("reason", ""),
            score=data.get("score", 0.0),
            related_memory_ids=data.get("related_memory_ids", []),
        )


@dataclass
class LearningStatus:
    """Status of the learning system."""
    enabled: bool = True
    last_analysis: Optional[str] = None
    next_scheduled: Optional[str] = None
    total_analyses: int = 0
    relationships_created: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningStatus":
        return cls(
            enabled=data.get("enabled", True),
            last_analysis=data.get("last_analysis"),
            next_scheduled=data.get("next_scheduled"),
            total_analyses=data.get("total_analyses", 0),
            relationships_created=data.get("relationships_created", 0),
        )


@dataclass
class LearningMetrics:
    """Metrics from the learning system."""
    total_memories: int = 0
    total_relationships: int = 0
    avg_relationships_per_memory: float = 0.0
    cluster_count: int = 0
    orphan_count: int = 0
    stale_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningMetrics":
        return cls(
            total_memories=data.get("total_memories", 0),
            total_relationships=data.get("total_relationships", 0),
            avg_relationships_per_memory=data.get("avg_relationships_per_memory", 0.0),
            cluster_count=data.get("cluster_count", 0),
            orphan_count=data.get("orphan_count", 0),
            stale_count=data.get("stale_count", 0),
        )


# ============================================================================
# LEARNING API
# ============================================================================

class LearningAPI:
    """
    Metacognitive learning analysis and self-improvement.

    Features:
    - On-demand learning analysis
    - Relationship suggestion detection
    - Weight adjustment recommendations
    - Maintenance suggestions (archive, merge, quality)
    - Enhanced pattern analysis

    Example:
        from recallbricks_langchain import RecallBricksClient, LearningAPI

        client = RecallBricksClient(api_key="your-key")
        learning = LearningAPI(client)

        # Run analysis
        result = learning.analyze()
        print(f"Found {len(result.relationship_suggestions)} new relationship suggestions")

        # Apply suggestions automatically
        learning.analyze(auto_apply=True)

        # Get maintenance suggestions
        suggestions = learning.get_maintenance_suggestions()
        for s in suggestions:
            if s.suggestion_type == "archive":
                print(f"Consider archiving: {s.memory_id} - {s.reason}")
    """

    def __init__(self, client: RecallBricksClient):
        self.client = client

    def analyze(
        self,
        auto_apply: bool = False,
        user_id: str = None,
    ) -> LearningAnalysisResult:
        """
        Run on-demand learning analysis.

        Analyzes memory access patterns, detects clusters,
        and suggests new relationships.

        Args:
            auto_apply: Automatically apply relationship suggestions
            user_id: User ID (required for service token auth)

        Returns:
            LearningAnalysisResult with suggestions and adjustments
        """
        payload = {"auto_apply": auto_apply}
        result = self.client.post("/learning/analyze", payload, user_id=user_id)
        return LearningAnalysisResult.from_dict(result.get("result", result))

    def analyze_enhanced(
        self,
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Run enhanced pattern analysis.

        More comprehensive analysis including anomaly detection
        and advanced pattern recognition.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Enhanced analysis results
        """
        return self.client.post("/learning/analyze-enhanced", {}, user_id=user_id)

    def apply_suggestions(
        self,
        suggestion_ids: List[str] = None,
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Apply relationship suggestions from analysis.

        Args:
            suggestion_ids: Specific suggestions to apply (or all if None)
            user_id: User ID (required for service token auth)

        Returns:
            Result of applying suggestions
        """
        payload = {}
        if suggestion_ids:
            payload["suggestion_ids"] = suggestion_ids

        return self.client.post("/learning/apply-suggestions", payload, user_id=user_id)

    def get_status(self, user_id: str = None) -> LearningStatus:
        """
        Get learning system status.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            LearningStatus with system state
        """
        result = self.client.get("/learning/status", user_id=user_id)
        return LearningStatus.from_dict(result)

    def get_metrics(self, user_id: str = None) -> LearningMetrics:
        """
        Get learning system metrics.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            LearningMetrics with system statistics
        """
        result = self.client.get("/learning/metrics", user_id=user_id)
        return LearningMetrics.from_dict(result)

    def get_maintenance_suggestions(
        self,
        user_id: str = None,
    ) -> List[MaintenanceSuggestion]:
        """
        Get memory maintenance suggestions.

        Returns suggestions for:
        - Archive: Memories that haven't been accessed in a long time
        - Merge: Duplicate or highly similar memories
        - Quality: Memories with low quality scores

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            List of MaintenanceSuggestion objects
        """
        result = self.client.get("/learning/maintenance-suggestions", user_id=user_id)
        suggestions = result.get("suggestions", result.get("items", []))
        return [MaintenanceSuggestion.from_dict(s) for s in suggestions]
