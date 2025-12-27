"""RecallBricks Agents API

Agent context injection and Identity Fusion Protocol.
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
class AgentProfile:
    """Full agent identity profile for context injection."""
    agent_id: str
    agent_name: str
    purpose: Optional[str] = None
    expertise_domains: List[str] = field(default_factory=list)
    personality_traits: List[str] = field(default_factory=list)
    communication_style: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentProfile":
        return cls(
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            purpose=data.get("purpose"),
            expertise_domains=data.get("expertise_domains", []),
            personality_traits=data.get("personality_traits", []),
            communication_style=data.get("communication_style"),
            constraints=data.get("constraints", []),
            goals=data.get("goals", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemorySummary:
    """Summary of a memory for context injection."""
    id: str
    text: str
    importance: str = "medium"
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySummary":
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            importance=data.get("importance", "medium"),
            created_at=data.get("created_at"),
        )


@dataclass
class AgentContext:
    """Context prepared for agent injection."""
    recent_memories: List[MemorySummary] = field(default_factory=list)
    important_memories: List[MemorySummary] = field(default_factory=list)
    relationship_graph: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentContext":
        recent = [MemorySummary.from_dict(m) for m in data.get("recent_memories", [])]
        important = [MemorySummary.from_dict(m) for m in data.get("important_memories", [])]
        return cls(
            recent_memories=recent,
            important_memories=important,
            relationship_graph=data.get("relationship_graph", {}),
        )


@dataclass
class ContextInjectionResult:
    """Result from agent context injection."""
    agent_id: str
    agent_profile: AgentProfile
    context: AgentContext
    system_prompt: str
    processing_time_ms: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextInjectionResult":
        profile = AgentProfile.from_dict(data.get("agent_profile", {}))
        context = AgentContext.from_dict(data.get("context", {}))
        return cls(
            agent_id=data.get("agent_id", ""),
            agent_profile=profile,
            context=context,
            system_prompt=data.get("system_prompt", ""),
            processing_time_ms=data.get("processing_time_ms", 0),
        )


@dataclass
class IdentityValidation:
    """Result from identity validation."""
    valid: bool
    agent_id: str
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityValidation":
        return cls(
            valid=data.get("valid", False),
            agent_id=data.get("agent_id", ""),
            missing_fields=data.get("missing_fields", []),
            warnings=data.get("warnings", []),
        )


# ============================================================================
# AGENTS API
# ============================================================================

class AgentsAPI:
    """
    Agent context injection and Identity Fusion Protocol.

    Features:
    - Context injection with configurable depth
    - Identity Fusion Protocol (flexible identity schema)
    - Pre-built system prompts
    - Relationship graph context

    Context Depths:
    - quick: Recent memories only (~50ms)
    - standard: Recent + important memories (~200ms)
    - comprehensive: Full context with relationships (~500ms)

    Example:
        from recallbricks_langchain import RecallBricksClient, AgentsAPI

        client = RecallBricksClient(api_key="your-key")
        agents = AgentsAPI(client)

        # Get context for agent injection
        result = agents.get_context(
            agent_id="my-agent",
            depth="comprehensive"
        )

        # Use system prompt in your LLM call
        messages = [
            {"role": "system", "content": result.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Access memory context
        for memory in result.context.important_memories:
            print(f"Important: {memory.text}")
    """

    def __init__(self, client: RecallBricksClient):
        self.client = client

    def get_context(
        self,
        agent_id: str,
        depth: str = "standard",
        user_id: str = None,
    ) -> ContextInjectionResult:
        """
        Get context for agent injection.

        Retrieves agent profile, relevant memories, and generates
        a system prompt for LLM context.

        Args:
            agent_id: Agent UUID or identifier
            depth: Context depth ('quick', 'standard', 'comprehensive')
            user_id: User ID for context (default: 'default')

        Returns:
            ContextInjectionResult with profile, context, and system prompt

        Depth Options:
        - quick: Only recent memories, fastest (~50ms)
        - standard: Recent + important memories (~200ms)
        - comprehensive: Full context with relationship graph (~500ms)
        """
        if not agent_id:
            raise ValidationError("agent_id is required")
        if depth not in ["quick", "standard", "comprehensive"]:
            raise ValidationError("depth must be 'quick', 'standard', or 'comprehensive'")

        payload = {
            "depth": depth,
        }
        if user_id:
            payload["user_id"] = user_id

        result = self.client.post(
            f"/agents/{agent_id}/context",
            payload,
            user_id=user_id
        )
        return ContextInjectionResult.from_dict(result)

    def validate_identity(
        self,
        identity: Dict[str, Any],
        user_id: str = None,
    ) -> IdentityValidation:
        """
        Validate agent identity against Identity Fusion Protocol.

        Checks that the identity schema is valid and complete.

        Args:
            identity: Agent identity dictionary
            user_id: User ID (required for service token auth)

        Returns:
            IdentityValidation with validation results

        Identity Schema (flexible):
        - agent_id: Required unique identifier
        - agent_name: Required display name
        - purpose: Optional description of agent purpose
        - expertise_domains: Optional list of expertise areas
        - personality_traits: Optional personality descriptors
        - communication_style: Optional style description
        - constraints: Optional list of constraints
        - goals: Optional list of goals
        """
        if not identity:
            raise ValidationError("identity is required")

        result = self.client.post(
            "/agents/validate-identity",
            {"identity": identity},
            user_id=user_id
        )
        return IdentityValidation.from_dict(result)

    def create_system_prompt(
        self,
        agent_profile: AgentProfile,
        context: AgentContext = None,
        include_memories: bool = True,
        max_memories: int = 10,
    ) -> str:
        """
        Create a system prompt from agent profile and context.

        This is a client-side helper for building system prompts.
        For server-side generation, use get_context().

        Args:
            agent_profile: Agent identity profile
            context: Optional memory context
            include_memories: Include memory context in prompt
            max_memories: Maximum memories to include

        Returns:
            Generated system prompt string
        """
        parts = []

        # Agent identity
        parts.append(f"You are {agent_profile.agent_name}.")

        if agent_profile.purpose:
            parts.append(f"Purpose: {agent_profile.purpose}")

        if agent_profile.expertise_domains:
            domains = ", ".join(agent_profile.expertise_domains)
            parts.append(f"Expertise: {domains}")

        if agent_profile.personality_traits:
            traits = ", ".join(agent_profile.personality_traits)
            parts.append(f"Personality: {traits}")

        if agent_profile.communication_style:
            parts.append(f"Communication Style: {agent_profile.communication_style}")

        if agent_profile.constraints:
            parts.append("Constraints:")
            for c in agent_profile.constraints:
                parts.append(f"- {c}")

        if agent_profile.goals:
            parts.append("Goals:")
            for g in agent_profile.goals:
                parts.append(f"- {g}")

        # Memory context
        if include_memories and context:
            memories_added = 0

            if context.important_memories:
                parts.append("\nImportant Context:")
                for mem in context.important_memories[:max_memories]:
                    parts.append(f"- {mem.text}")
                    memories_added += 1

            remaining = max_memories - memories_added
            if remaining > 0 and context.recent_memories:
                parts.append("\nRecent Context:")
                for mem in context.recent_memories[:remaining]:
                    parts.append(f"- {mem.text}")

        return "\n".join(parts)
