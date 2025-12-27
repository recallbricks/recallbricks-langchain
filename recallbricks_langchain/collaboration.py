"""RecallBricks Collaboration API

Multi-agent collaboration, knowledge synthesis, and conflict resolution.
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
class Agent:
    """AI agent profile."""
    id: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    reputation_score: float = 0.5
    contribution_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        return cls(
            id=data.get("id", data.get("agent_id", "")),
            name=data.get("name", data.get("agent_name", "")),
            capabilities=data.get("capabilities", []),
            reputation_score=data.get("reputation_score", 0.5),
            contribution_count=data.get("contribution_count", 0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
        )


@dataclass
class AgentPerformance:
    """Agent performance metrics."""
    agent_id: str
    total_contributions: int = 0
    accepted_contributions: int = 0
    rejected_contributions: int = 0
    avg_confidence: float = 0.0
    reputation_trend: str = "stable"  # improving, stable, declining

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPerformance":
        return cls(
            agent_id=data.get("agent_id", ""),
            total_contributions=data.get("total_contributions", 0),
            accepted_contributions=data.get("accepted_contributions", 0),
            rejected_contributions=data.get("rejected_contributions", 0),
            avg_confidence=data.get("avg_confidence", 0.0),
            reputation_trend=data.get("reputation_trend", "stable"),
        )


@dataclass
class Contribution:
    """Agent memory contribution."""
    id: str
    memory_id: str
    agent_id: str
    contribution_type: str  # create, update, validate
    confidence: float = 0.5
    validation_status: str = "pending"  # pending, accepted, rejected
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contribution":
        return cls(
            id=data.get("id", ""),
            memory_id=data.get("memory_id", ""),
            agent_id=data.get("agent_id", ""),
            contribution_type=data.get("contribution_type", "create"),
            confidence=data.get("confidence", 0.5),
            validation_status=data.get("validation_status", "pending"),
            created_at=data.get("created_at"),
        )


@dataclass
class Conflict:
    """Memory conflict between agents."""
    id: str
    memory_id_1: str
    memory_id_2: str
    conflict_type: str  # contradiction, duplicate, outdated
    severity: str = "low"  # low, medium, high
    description: Optional[str] = None
    resolution_status: str = "unresolved"  # unresolved, resolved, dismissed
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conflict":
        return cls(
            id=data.get("id", ""),
            memory_id_1=data.get("memory_id_1", ""),
            memory_id_2=data.get("memory_id_2", ""),
            conflict_type=data.get("conflict_type", ""),
            severity=data.get("severity", "low"),
            description=data.get("description"),
            resolution_status=data.get("resolution_status", "unresolved"),
            created_at=data.get("created_at"),
        )


@dataclass
class SynthesisResult:
    """Result from knowledge synthesis."""
    synthesized_text: str
    source_memory_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    agent_contributions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesisResult":
        return cls(
            synthesized_text=data.get("synthesized_text", data.get("text", "")),
            source_memory_ids=data.get("source_memory_ids", []),
            confidence=data.get("confidence", 0.0),
            agent_contributions=data.get("agent_contributions", []),
        )


# ============================================================================
# COLLABORATION API
# ============================================================================

class CollaborationAPI:
    """
    Multi-agent collaboration and knowledge synthesis.

    Features:
    - Register and manage AI agents
    - Track agent contributions and reputation
    - Detect and resolve memory conflicts
    - Synthesize knowledge from multiple sources
    - Share learning across agents

    Example:
        from recallbricks_langchain import RecallBricksClient, CollaborationAPI

        client = RecallBricksClient(api_key="your-key")
        collab = CollaborationAPI(client)

        # Register an agent
        agent = collab.register_agent(
            name="Research Assistant",
            capabilities=["summarization", "research", "fact-checking"]
        )

        # Contribute memory as agent
        contribution = collab.contribute(
            agent_id=agent.id,
            text="Important finding from research",
            confidence=0.9
        )

        # Detect conflicts
        conflicts = collab.detect_conflicts()
        for conflict in conflicts:
            print(f"Conflict: {conflict.conflict_type} - {conflict.description}")
    """

    def __init__(self, client: RecallBricksClient):
        self.client = client

    # =========================================================================
    # Agent Management
    # =========================================================================

    def register_agent(
        self,
        name: str,
        capabilities: List[str] = None,
        metadata: Dict[str, Any] = None,
        user_id: str = None,
    ) -> Agent:
        """
        Register a new AI agent.

        Args:
            name: Agent name
            capabilities: List of agent capabilities
            metadata: Optional metadata (model, version, etc.)
            user_id: User ID (required for service token auth)

        Returns:
            Registered Agent object
        """
        if not name:
            raise ValidationError("name is required")

        payload = {"name": name}
        if capabilities:
            payload["capabilities"] = capabilities
        if metadata:
            payload["metadata"] = metadata

        result = self.client.post("/collaboration/agents", payload, user_id=user_id)
        return Agent.from_dict(result)

    def list_agents(self, user_id: str = None) -> List[Agent]:
        """
        List all registered agents.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            List of Agent objects
        """
        result = self.client.get("/collaboration/agents", user_id=user_id)
        return [Agent.from_dict(a) for a in result.get("agents", result.get("items", []))]

    def get_agent(self, agent_id: str, user_id: str = None) -> Agent:
        """
        Get agent profile by ID.

        Args:
            agent_id: Agent UUID
            user_id: User ID (required for service token auth)

        Returns:
            Agent object
        """
        if not agent_id:
            raise ValidationError("agent_id is required")

        result = self.client.get(f"/collaboration/agents/{agent_id}", user_id=user_id)
        return Agent.from_dict(result)

    def get_agent_performance(
        self,
        agent_id: str,
        user_id: str = None,
    ) -> AgentPerformance:
        """
        Get agent performance metrics.

        Args:
            agent_id: Agent UUID
            user_id: User ID (required for service token auth)

        Returns:
            AgentPerformance metrics
        """
        if not agent_id:
            raise ValidationError("agent_id is required")

        result = self.client.get(
            f"/collaboration/agents/{agent_id}/performance",
            user_id=user_id
        )
        return AgentPerformance.from_dict(result)

    def recalculate_reputation(
        self,
        agent_id: str,
        user_id: str = None,
    ) -> Agent:
        """
        Recalculate agent reputation score.

        Args:
            agent_id: Agent UUID
            user_id: User ID (required for service token auth)

        Returns:
            Updated Agent object
        """
        if not agent_id:
            raise ValidationError("agent_id is required")

        result = self.client.post(
            f"/collaboration/agents/{agent_id}/recalculate-reputation",
            {},
            user_id=user_id
        )
        return Agent.from_dict(result)

    # =========================================================================
    # Contributions
    # =========================================================================

    def contribute(
        self,
        agent_id: str,
        text: str,
        confidence: float = 0.5,
        source: str = "agent",
        metadata: Dict[str, Any] = None,
        user_id: str = None,
    ) -> Contribution:
        """
        Agent contributes a memory.

        Args:
            agent_id: Agent UUID
            text: Memory text content
            confidence: Confidence level (0.0-1.0)
            source: Source identifier
            metadata: Optional metadata
            user_id: User ID (required for service token auth)

        Returns:
            Contribution object
        """
        if not agent_id:
            raise ValidationError("agent_id is required")
        if not text:
            raise ValidationError("text is required")

        payload = {
            "agent_id": agent_id,
            "text": text,
            "confidence": confidence,
            "source": source,
        }
        if metadata:
            payload["metadata"] = metadata

        result = self.client.post("/collaboration/contribute", payload, user_id=user_id)
        return Contribution.from_dict(result)

    def validate_contribution(
        self,
        contribution_id: str,
        status: str,
        reason: str = None,
        user_id: str = None,
    ) -> Contribution:
        """
        Validate or reject a contribution.

        Args:
            contribution_id: Contribution UUID
            status: 'accepted' or 'rejected'
            reason: Optional reason for decision
            user_id: User ID (required for service token auth)

        Returns:
            Updated Contribution object
        """
        if not contribution_id:
            raise ValidationError("contribution_id is required")
        if status not in ["accepted", "rejected"]:
            raise ValidationError("status must be 'accepted' or 'rejected'")

        payload = {"status": status}
        if reason:
            payload["reason"] = reason

        result = self.client.post(
            f"/collaboration/contributions/{contribution_id}/validate",
            payload,
            user_id=user_id
        )
        return Contribution.from_dict(result)

    # =========================================================================
    # Knowledge Synthesis
    # =========================================================================

    def synthesize(
        self,
        memory_ids: List[str],
        prompt: str = None,
        user_id: str = None,
    ) -> SynthesisResult:
        """
        Synthesize knowledge from multiple memories.

        Combines insights from multiple memories into a coherent synthesis.

        Args:
            memory_ids: List of memory UUIDs to synthesize
            prompt: Optional synthesis prompt/instruction
            user_id: User ID (required for service token auth)

        Returns:
            SynthesisResult with combined knowledge
        """
        if not memory_ids or len(memory_ids) < 2:
            raise ValidationError("At least 2 memory_ids required for synthesis")

        payload = {"memory_ids": memory_ids}
        if prompt:
            payload["prompt"] = prompt

        result = self.client.post("/collaboration/synthesize", payload, user_id=user_id)
        return SynthesisResult.from_dict(result)

    # =========================================================================
    # Conflict Detection and Resolution
    # =========================================================================

    def detect_conflicts(
        self,
        memory_ids: List[str] = None,
        user_id: str = None,
    ) -> List[Conflict]:
        """
        Detect memory conflicts.

        Identifies contradictions, duplicates, and outdated information.

        Args:
            memory_ids: Optional specific memories to check
            user_id: User ID (required for service token auth)

        Returns:
            List of Conflict objects
        """
        payload = {}
        if memory_ids:
            payload["memory_ids"] = memory_ids

        result = self.client.post("/collaboration/detect-conflicts", payload, user_id=user_id)
        return [Conflict.from_dict(c) for c in result.get("conflicts", [])]

    def list_conflicts(
        self,
        status: str = None,
        user_id: str = None,
    ) -> List[Conflict]:
        """
        List all conflicts.

        Args:
            status: Filter by status ('unresolved', 'resolved', 'dismissed')
            user_id: User ID (required for service token auth)

        Returns:
            List of Conflict objects
        """
        params = {}
        if status:
            params["status"] = status

        result = self.client.get("/collaboration/conflicts", params=params, user_id=user_id)
        return [Conflict.from_dict(c) for c in result.get("conflicts", result.get("items", []))]

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        keep_memory_id: str = None,
        user_id: str = None,
    ) -> Conflict:
        """
        Resolve a conflict.

        Args:
            conflict_id: Conflict UUID
            resolution: Resolution type ('keep_first', 'keep_second', 'merge', 'dismiss')
            keep_memory_id: Memory ID to keep (for keep_first/keep_second)
            user_id: User ID (required for service token auth)

        Returns:
            Updated Conflict object
        """
        if not conflict_id:
            raise ValidationError("conflict_id is required")
        if not resolution:
            raise ValidationError("resolution is required")

        payload = {"resolution": resolution}
        if keep_memory_id:
            payload["keep_memory_id"] = keep_memory_id

        result = self.client.post(
            f"/collaboration/conflicts/{conflict_id}/resolve",
            payload,
            user_id=user_id
        )
        return Conflict.from_dict(result)

    # =========================================================================
    # Learning Sharing
    # =========================================================================

    def share_learning(
        self,
        from_agent_id: str,
        to_agent_ids: List[str],
        pattern_type: str = "all",
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Share learning patterns across agents.

        Args:
            from_agent_id: Source agent UUID
            to_agent_ids: Target agent UUIDs
            pattern_type: Type of patterns to share ('all', 'relationships', 'weights')
            user_id: User ID (required for service token auth)

        Returns:
            Result of sharing operation
        """
        if not from_agent_id:
            raise ValidationError("from_agent_id is required")
        if not to_agent_ids:
            raise ValidationError("to_agent_ids is required")

        payload = {
            "from_agent_id": from_agent_id,
            "to_agent_ids": to_agent_ids,
            "pattern_type": pattern_type,
        }

        return self.client.post("/collaboration/share-learning", payload, user_id=user_id)

    def get_dashboard(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get collaboration system dashboard.

        Returns overview of agents, contributions, conflicts, and activity.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Dashboard data dictionary
        """
        return self.client.get("/collaboration/dashboard", user_id=user_id)
