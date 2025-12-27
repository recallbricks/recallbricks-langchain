"""RecallBricks Relationships API

Semantic relationship detection and graph traversal between memories.
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
class RelatedMemory:
    """Summary of a related memory."""
    id: str
    text: str
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelatedMemory":
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            created_at=data.get("created_at"),
        )


@dataclass
class Relationship:
    """Relationship between two memories."""
    id: str
    from_memory_id: str
    to_memory_id: str
    type: str  # related_to, caused_by, similar_to, follows, contradicts
    strength: float  # 0.0 to 1.0
    explanation: Optional[str] = None
    created_at: Optional[str] = None
    related_memory: Optional[RelatedMemory] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        related = None
        if "related_memory" in data:
            related = RelatedMemory.from_dict(data["related_memory"])

        return cls(
            id=data.get("id", ""),
            from_memory_id=data.get("from_memory_id", data.get("memory_id", "")),
            to_memory_id=data.get("to_memory_id", data.get("related_memory_id", "")),
            type=data.get("type", data.get("relationship_type", "related_to")),
            strength=data.get("strength", 0.5),
            explanation=data.get("explanation"),
            created_at=data.get("created_at"),
            related_memory=related,
        )


@dataclass
class RelationshipGraph:
    """Graph of relationships for BFS traversal."""
    memory_id: str
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipGraph":
        return cls(
            memory_id=data.get("memoryId", data.get("memory_id", "")),
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
            depth=data.get("depth", 0),
        )


@dataclass
class RelationshipTypeStats:
    """Statistics for relationship types."""
    type: str
    count: int
    avg_strength: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipTypeStats":
        return cls(
            type=data.get("type", ""),
            count=data.get("count", 0),
            avg_strength=data.get("avg_strength", 0.0),
        )


# ============================================================================
# RELATIONSHIPS API
# ============================================================================

class RelationshipsAPI:
    """
    Semantic relationship operations between memories.

    Features:
    - Get relationships for a memory
    - Graph traversal (BFS) for relationship chains
    - Relationship type statistics
    - Delete relationships

    Relationship Types:
    - related_to: General semantic relationship
    - caused_by: Causal relationship (A caused B)
    - similar_to: High semantic similarity
    - follows: Sequential relationship (A comes before B)
    - contradicts: Contradictory information

    Example:
        from recallbricks_langchain import RecallBricksClient, RelationshipsAPI

        client = RecallBricksClient(api_key="your-key")
        relationships = RelationshipsAPI(client)

        # Get all relationships for a memory
        rels = relationships.get_for_memory("memory-id")
        for rel in rels:
            print(f"{rel.type}: {rel.strength:.2f} - {rel.explanation}")

        # Get relationship graph
        graph = relationships.get_graph("memory-id", depth=2)
        print(f"Found {len(graph.nodes)} connected memories")
    """

    def __init__(self, client: RecallBricksClient):
        self.client = client

    def get_for_memory(
        self,
        memory_id: str,
        type: str = None,
        min_strength: float = None,
        limit: int = 50,
        user_id: str = None,
    ) -> List[Relationship]:
        """
        Get all relationships for a memory.

        Args:
            memory_id: Memory UUID
            type: Filter by relationship type
            min_strength: Minimum strength threshold (0.0-1.0)
            limit: Maximum results (default: 50)
            user_id: User ID (required for service token auth)

        Returns:
            List of Relationship objects
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        params = {"limit": limit}
        if type:
            params["type"] = type
        if min_strength is not None:
            params["minStrength"] = min_strength

        result = self.client.get(
            f"/relationships/memory/{memory_id}",
            params=params,
            user_id=user_id
        )
        return [Relationship.from_dict(r) for r in result.get("relationships", [])]

    def get_graph(
        self,
        memory_id: str,
        depth: int = 2,
        user_id: str = None,
    ) -> RelationshipGraph:
        """
        Get relationship graph using BFS traversal.

        Traverses relationships to find connected memories up to
        the specified depth.

        Args:
            memory_id: Starting memory UUID
            depth: Traversal depth (default: 2)
            user_id: User ID (required for service token auth)

        Returns:
            RelationshipGraph with nodes and edges
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        params = {"depth": depth}
        result = self.client.get(
            f"/relationships/graph/{memory_id}",
            params=params,
            user_id=user_id
        )
        return RelationshipGraph.from_dict(result)

    def get_type_stats(self, user_id: str = None) -> List[RelationshipTypeStats]:
        """
        Get relationship type statistics.

        Returns counts and average strengths for each relationship type.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            List of RelationshipTypeStats
        """
        result = self.client.get("/relationships/types", user_id=user_id)
        return [RelationshipTypeStats.from_dict(s) for s in result.get("types", result.get("stats", []))]

    def delete(self, relationship_id: str, user_id: str = None) -> bool:
        """
        Delete a specific relationship.

        Args:
            relationship_id: Relationship UUID
            user_id: User ID (required for service token auth)

        Returns:
            True if deleted successfully
        """
        if not relationship_id:
            raise ValidationError("relationship_id is required")

        self.client.delete(f"/relationships/{relationship_id}", user_id=user_id)
        return True

    def health(self, user_id: str = None) -> Dict[str, Any]:
        """
        Check relationship detection service health.

        Args:
            user_id: User ID (required for service token auth)

        Returns:
            Health status dictionary
        """
        return self.client.get("/relationships/health", user_id=user_id)
