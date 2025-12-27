"""RecallBricks Memories API

Complete memory CRUD operations with automatic metadata extraction,
semantic search, and intelligent context retrieval.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .client import (
    RecallBricksClient,
    RecallBricksError,
    ValidationError,
)


logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Memory:
    """Memory object from RecallBricks API."""
    id: str
    text: str
    source: str = "api"
    project_id: str = "default"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    usage_count: int = 0
    helpfulness_score: float = 0.5
    user_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            source=data.get("source", "api"),
            project_id=data.get("project_id", "default"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            usage_count=data.get("usage_count", 0),
            helpfulness_score=data.get("helpfulness_score", 0.5),
            user_id=data.get("user_id"),
        )


@dataclass
class ExtractedMetadata:
    """Metadata extracted by learn() endpoint."""
    category: str = "General"
    subcategory: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    importance: str = "medium"  # low, medium, high, critical
    sentiment: str = "neutral"  # positive, neutral, negative
    actionable: bool = False
    summary: Optional[str] = None
    confidence: float = 0.8

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedMetadata":
        return cls(
            category=data.get("category", "General"),
            subcategory=data.get("subcategory"),
            tags=data.get("tags", []),
            entities=data.get("entities", []),
            importance=data.get("importance", "medium"),
            sentiment=data.get("sentiment", "neutral"),
            actionable=data.get("actionable", False),
            summary=data.get("summary"),
            confidence=data.get("confidence", 0.8),
        )


@dataclass
class LearnResult:
    """Result from learn() endpoint with extracted metadata."""
    memory: Memory
    extracted_metadata: ExtractedMetadata
    extraction_tier: int = 1
    extraction_method: str = "heuristic"  # heuristic, llm, null
    metadata_version: str = "1.1.0"
    processing: bool = False
    processing_time_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnResult":
        return cls(
            memory=Memory.from_dict(data),
            extracted_metadata=ExtractedMetadata.from_dict(data.get("extracted_metadata", {})),
            extraction_tier=data.get("extraction_tier", 1),
            extraction_method=data.get("extraction_method", "heuristic"),
            metadata_version=data.get("metadata_version", "1.1.0"),
            processing=data.get("processing", False),
            processing_time_ms=data.get("processing_time_ms"),
        )


@dataclass
class SearchResult:
    """Memory with search scoring information."""
    memory: Memory
    base_similarity: float = 0.0
    weighted_score: float = 0.0
    boosted_by_usage: bool = False
    boosted_by_recency: bool = False
    penalized_by_age: bool = False
    access_frequency: str = "unused"  # high, medium, low, unused

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        return cls(
            memory=Memory.from_dict(data),
            base_similarity=data.get("base_similarity", data.get("score", 0.0)),
            weighted_score=data.get("weighted_score", data.get("score", 0.0)),
            boosted_by_usage=data.get("boosted_by_usage", False),
            boosted_by_recency=data.get("boosted_by_recency", False),
            penalized_by_age=data.get("penalized_by_age", False),
            access_frequency=data.get("access_frequency", "unused"),
        )


@dataclass
class RecallResult:
    """Result from recall() endpoint with organized categories."""
    memories: List[SearchResult]
    categories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    count: int = 0
    query: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecallResult":
        memories = [SearchResult.from_dict(m) for m in data.get("memories", [])]
        return cls(
            memories=memories,
            categories=data.get("categories", {}),
            count=data.get("count", len(memories)),
            query=data.get("query", ""),
        )


@dataclass
class ContextResult:
    """Result from intelligent context retrieval."""
    query: str
    keywords_extracted: List[str] = field(default_factory=list)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    count: int = 0
    cross_llm: bool = False
    llm: Optional[str] = None
    intelligence: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextResult":
        return cls(
            query=data.get("query", ""),
            keywords_extracted=data.get("keywords_extracted", []),
            memories=data.get("memories", []),
            count=data.get("count", 0),
            cross_llm=data.get("crossLLM", False),
            llm=data.get("llm"),
            intelligence=data.get("intelligence", {}),
        )


# ============================================================================
# MEMORIES API
# ============================================================================

class MemoriesAPI:
    """
    Complete Memory CRUD operations with automatic metadata extraction.

    Features:
    - Create, read, update, delete memories
    - Smart learning with automatic metadata extraction (Tier 1-3)
    - Semantic search with usage/recency weighting
    - Organized recall with category summaries
    - Intelligent context retrieval with keyword extraction
    - Helpfulness feedback for learning
    - Batch operations

    Example:
        from recallbricks_langchain import RecallBricksClient, MemoriesAPI

        client = RecallBricksClient(api_key="your-key")
        memories = MemoriesAPI(client)

        # Smart save with auto-extraction
        result = memories.learn("User prefers dark mode for all applications")
        print(result.extracted_metadata.tags)  # ['preferences', 'ui', 'dark-mode']

        # Semantic search with weighting
        results = memories.search("user preferences", weight_by_usage=True)

        # Organized recall for LLM context
        recall = memories.recall("what does the user prefer?", organized=True)
        for cat, summary in recall.categories.items():
            print(f"{cat}: {summary['summary']}")
    """

    def __init__(self, client: RecallBricksClient):
        """
        Initialize Memories API.

        Args:
            client: RecallBricksClient instance
        """
        self.client = client

    def _validate_text(self, text: str, max_length: int = 10000) -> str:
        """Validate and sanitize text input."""
        if not text or not isinstance(text, str):
            raise ValidationError("text must be a non-empty string")
        text = text.replace('\x00', '')  # Remove null bytes
        if len(text) > max_length:
            raise ValidationError(f"text exceeds maximum length of {max_length} characters")
        return text

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create(
        self,
        text: str,
        source: str = "api",
        project_id: str = "default",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        agent_id: str = None,
        raw: bool = False,
        user_id: str = None,
    ) -> Memory:
        """
        Create a memory with embedding generation.

        This is the basic create operation. For automatic metadata extraction,
        use learn() instead.

        Args:
            text: Memory text content (max 10,000 chars)
            source: Source identifier (default: 'api')
            project_id: Project for multi-tenant isolation (default: 'default')
            tags: Optional list of tags
            metadata: Optional custom metadata
            agent_id: Optional agent ID for contribution tracking
            raw: Skip all processing if True
            user_id: User ID (required for service token auth)

        Returns:
            Created Memory object
        """
        text = self._validate_text(text)

        payload = {
            "text": text,
            "source": source,
            "project_id": project_id,
        }

        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata
        if agent_id:
            payload["agent_id"] = agent_id
        if raw:
            payload["raw"] = True

        result = self.client.post("/memories", payload, user_id=user_id)
        return Memory.from_dict(result)

    def get(self, memory_id: str, user_id: str = None) -> Memory:
        """
        Get a single memory by ID.

        Args:
            memory_id: Memory UUID
            user_id: User ID (required for service token auth)

        Returns:
            Memory object
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        result = self.client.get(f"/memories/{memory_id}", user_id=user_id)
        return Memory.from_dict(result)

    def list(
        self,
        limit: int = 10,
        offset: int = 0,
        project_id: str = None,
        source: str = None,
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        List memories with pagination.

        Args:
            limit: Number of results (1-100, default: 10)
            offset: Pagination offset (default: 0)
            project_id: Filter by project
            source: Filter by source
            user_id: User ID (required for service token auth)

        Returns:
            Dictionary with 'memories', 'count', 'total', 'limit', 'offset'
        """
        if limit < 1 or limit > 100:
            raise ValidationError("limit must be between 1 and 100")

        params = {"limit": limit, "offset": offset}
        if project_id:
            params["project_id"] = project_id
        if source:
            params["source"] = source

        result = self.client.get("/memories", params=params, user_id=user_id)

        return {
            "memories": [Memory.from_dict(m) for m in result.get("memories", result.get("items", []))],
            "count": result.get("count", 0),
            "total": result.get("total", 0),
            "limit": result.get("limit", limit),
            "offset": result.get("offset", offset),
        }

    def update(
        self,
        memory_id: str,
        text: str = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        user_id: str = None,
    ) -> Memory:
        """
        Update a memory.

        Args:
            memory_id: Memory UUID
            text: New text content
            tags: New tags
            metadata: New metadata
            user_id: User ID (required for service token auth)

        Returns:
            Updated Memory object
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        payload = {}
        if text is not None:
            payload["text"] = self._validate_text(text)
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata

        if not payload:
            raise ValidationError("At least one field must be provided for update")

        result = self.client.put(f"/memories/{memory_id}", payload, user_id=user_id)
        return Memory.from_dict(result)

    def delete(self, memory_id: str, user_id: str = None) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory UUID
            user_id: User ID (required for service token auth)

        Returns:
            True if deleted successfully
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        self.client.delete(f"/memories/{memory_id}", user_id=user_id)
        return True

    # =========================================================================
    # Smart Learning Operations
    # =========================================================================

    def learn(
        self,
        text: str,
        source: str = "api",
        project_id: str = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        tier: int = 1,
        sync: bool = False,
        raw: bool = False,
        skip_extraction: bool = False,
        context: str = None,
        agent_id: str = None,
        user_id: str = None,
    ) -> LearnResult:
        """
        Save memory with automatic metadata extraction.

        This is the recommended way to save memories. It automatically
        extracts tags, categories, entities, importance, and more.

        Extraction Tiers:
        - Tier 1: Heuristic extraction (<50ms, zero cost, ~80% quality)
        - Tier 2: Claude Haiku-4 extraction (~2s, token cost, 100% quality)
        - Tier 3: Claude Sonnet-4 extraction (enterprise, highest quality)

        Args:
            text: Memory text content
            source: Source identifier
            project_id: Project for multi-tenant isolation
            tags: Optional pre-defined tags
            metadata: Optional custom metadata
            tier: Extraction tier (1 or 2, default: 1)
            sync: Wait for LLM extraction to complete (tier 2)
            raw: Skip all processing
            skip_extraction: Skip metadata extraction
            context: Optional context to help extraction
            agent_id: Agent ID for contribution tracking
            user_id: User ID (required for service token auth)

        Returns:
            LearnResult with memory and extracted metadata

        Example:
            result = memories.learn(
                "The user mentioned they work at Google as a software engineer",
                tier=2,
                sync=True
            )
            print(result.extracted_metadata.entities)
            # [{'name': 'Google', 'type': 'organization'}]
        """
        text = self._validate_text(text)

        payload = {
            "text": text,
            "source": source,
        }

        if project_id:
            payload["project_id"] = project_id
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata
        if tier and tier != 1:
            payload["tier"] = tier
        if sync:
            payload["sync"] = True
        if raw:
            payload["raw"] = True
        if skip_extraction:
            payload["skip_extraction"] = True
        if context:
            payload["context"] = context
        if agent_id:
            payload["agent_id"] = agent_id

        result = self.client.post("/memories/learn", payload, user_id=user_id)
        return LearnResult.from_dict(result)

    def auto_save(
        self,
        text: str,
        source: str = "api",
        project_id: str = None,
        metadata: Dict[str, Any] = None,
        user_id: str = None,
    ) -> LearnResult:
        """
        Smart auto-save with importance classification.

        Automatically classifies importance and determines optimal storage.

        Args:
            text: Memory text content
            source: Source identifier
            project_id: Project for multi-tenant isolation
            metadata: Optional custom metadata
            user_id: User ID (required for service token auth)

        Returns:
            LearnResult with auto-classified metadata
        """
        text = self._validate_text(text)

        payload = {
            "text": text,
            "source": source,
        }

        if project_id:
            payload["project_id"] = project_id
        if metadata:
            payload["metadata"] = metadata

        result = self.client.post("/memories/auto-save", payload, user_id=user_id)
        return LearnResult.from_dict(result)

    def batch_create(
        self,
        memories: List[Dict[str, Any]],
        user_id: str = None,
    ) -> List[Memory]:
        """
        Batch create multiple memories.

        Args:
            memories: List of memory objects with 'text' and optional fields
            user_id: User ID (required for service token auth)

        Returns:
            List of created Memory objects

        Example:
            results = memories.batch_create([
                {"text": "First memory", "source": "import"},
                {"text": "Second memory", "source": "import", "tags": ["tag1"]},
            ])
        """
        if not memories or not isinstance(memories, list):
            raise ValidationError("memories must be a non-empty list")

        for m in memories:
            if "text" not in m:
                raise ValidationError("Each memory must have a 'text' field")
            m["text"] = self._validate_text(m["text"])

        result = self.client.post("/memories/batch", {"memories": memories}, user_id=user_id)
        return [Memory.from_dict(m) for m in result.get("memories", result.get("items", []))]

    # =========================================================================
    # Search and Recall Operations
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 10,
        weight_by_usage: bool = False,
        decay_old_memories: bool = False,
        learning_mode: bool = False,
        min_helpfulness_score: float = None,
        project_id: str = None,
        tags: List[str] = None,
        user_id: str = None,
    ) -> List[SearchResult]:
        """
        Semantic search with optional weighting.

        Args:
            query: Search query
            limit: Number of results (1-100, default: 10)
            weight_by_usage: Boost frequently used memories
            decay_old_memories: Penalize stale memories
            learning_mode: Enable learning mode
            min_helpfulness_score: Minimum helpfulness threshold
            project_id: Filter by project
            tags: Filter by tags
            user_id: User ID (required for service token auth)

        Returns:
            List of SearchResult objects with scoring information

        Weighting Algorithm (when weight_by_usage=True):
            weighted_score = base_similarity * (1 + log(usage_count + 1)) * helpfulness_score

        Decay Algorithm (when decay_old_memories=True):
            - Recent (<=7 days): +20% boost
            - Mid-range (7-90 days): No change
            - Stale (>=90 days): -30% penalty
        """
        if not query:
            raise ValidationError("query is required")
        if limit < 1 or limit > 100:
            raise ValidationError("limit must be between 1 and 100")

        payload = {
            "query": query,
            "limit": limit,
        }

        if weight_by_usage:
            payload["weight_by_usage"] = True
        if decay_old_memories:
            payload["decay_old_memories"] = True
        if learning_mode:
            payload["learning_mode"] = True
        if min_helpfulness_score is not None:
            payload["min_helpfulness_score"] = min_helpfulness_score
        if project_id:
            payload["project_id"] = project_id
        if tags:
            payload["tags"] = tags

        result = self.client.post("/memories/search", payload, user_id=user_id)
        return [SearchResult.from_dict(m) for m in result.get("memories", [])]

    def recall(
        self,
        query: str,
        limit: int = 10,
        organized: bool = True,
        project_id: str = None,
        user_id: str = None,
    ) -> RecallResult:
        """
        Recall memories with optional organized categories.

        When organized=True, returns category summaries for faster LLM
        context assembly (3-5x faster than flat lists).

        Args:
            query: Search query
            limit: Number of results (1-100, default: 10)
            organized: Include category summaries (default: True)
            project_id: Filter by project
            user_id: User ID (required for service token auth)

        Returns:
            RecallResult with memories and category summaries

        Example:
            result = memories.recall("user preferences", organized=True)

            # Category summaries for quick context
            for category, data in result.categories.items():
                print(f"{category}: {data['summary']} ({data['count']} memories)")

            # Detailed memories
            for sr in result.memories:
                print(f"- {sr.memory.text} (score: {sr.weighted_score})")
        """
        if not query:
            raise ValidationError("query is required")

        payload = {
            "query": query,
            "limit": limit,
            "organized": organized,
        }

        if project_id:
            payload["project_id"] = project_id

        result = self.client.post("/memories/recall", payload, user_id=user_id)
        return RecallResult.from_dict(result)

    def context(
        self,
        query: str,
        llm: str = None,
        limit: int = 10,
        project_id: str = None,
        conversation_history: List[str] = None,
        user_id: str = None,
    ) -> ContextResult:
        """
        Intelligent context retrieval with keyword extraction.

        Uses NLP to extract keywords from query and find most relevant
        memories for LLM context injection.

        Args:
            query: Search query
            llm: Optional LLM identifier for cross-LLM context
            limit: Number of results (default: 10)
            project_id: Filter by project
            conversation_history: Optional conversation context
            user_id: User ID (required for service token auth)

        Returns:
            ContextResult with extracted keywords and relevant memories
        """
        if not query:
            raise ValidationError("query is required")

        payload = {
            "query": query,
            "limit": limit,
        }

        if llm:
            payload["llm"] = llm
        if project_id:
            payload["project_id"] = project_id
        if conversation_history:
            payload["conversation_history"] = conversation_history

        result = self.client.post("/context", payload, user_id=user_id)
        return ContextResult.from_dict(result)

    def suggest(
        self,
        query: str = None,
        project_id: str = None,
        user_id: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Get context suggestions for a query.

        Args:
            query: Optional query for suggestions
            project_id: Filter by project
            user_id: User ID (required for service token auth)

        Returns:
            List of suggestion objects
        """
        payload = {}
        if query:
            payload["query"] = query
        if project_id:
            payload["project_id"] = project_id

        result = self.client.post("/memories/suggest", payload, user_id=user_id)
        return result.get("suggestions", [])

    # =========================================================================
    # Feedback and Learning Operations
    # =========================================================================

    def feedback(
        self,
        memory_id: str,
        helpful: bool,
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Submit helpfulness feedback for a memory.

        Feedback improves search ranking through the helpfulness score.
        Helpful memories are boosted, unhelpful ones are demoted.

        Args:
            memory_id: Memory UUID
            helpful: True if memory was helpful, False otherwise
            user_id: User ID (required for service token auth)

        Returns:
            Updated helpfulness information
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        payload = {"helpful": helpful}
        return self.client.post(f"/memories/{memory_id}/feedback", payload, user_id=user_id)

    def upgrade(
        self,
        memory_id: str,
        user_id: str = None,
    ) -> LearnResult:
        """
        Manually trigger tier upgrade for a memory.

        Forces upgrade from Tier 1 to Tier 2 (LLM extraction).
        Normally happens automatically based on usage (Hebbian learning).

        Args:
            memory_id: Memory UUID
            user_id: User ID (required for service token auth)

        Returns:
            LearnResult with upgraded metadata
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        result = self.client.post(f"/memories/{memory_id}/upgrade", {}, user_id=user_id)
        return LearnResult.from_dict(result)
