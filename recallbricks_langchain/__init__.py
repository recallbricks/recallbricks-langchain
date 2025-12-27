"""RecallBricks LangChain Integration v2.0

Official LangChain integration for RecallBricks Memory Graph API.

This SDK provides complete access to the RecallBricks API with:
- Full memory CRUD operations with automatic metadata extraction
- Semantic search with usage/recency weighting
- Organized recall with category summaries
- Relationship detection and graph traversal
- Metacognitive learning and self-improvement
- Multi-agent collaboration and conflict resolution
- Usage tracking and billing
- Comprehensive monitoring and observability

LangChain Integrations:
- RecallBricksMemory: Persistent memory for agents
- RecallBricksChatMessageHistory: Persistent chat history
- RecallBricksRetriever: RAG retriever with organized results

Enterprise-grade reliability:
- Connection pooling for performance
- Automatic retry with exponential backoff
- Circuit breaker for fault tolerance
- Rate limiting (client and server)
- Request deduplication
- Distributed tracing with request IDs
- Prometheus metrics export
- Health checks
- Graceful shutdown

Authentication:
- API Key (X-API-Key) for single-tenant applications
- Service Token (X-Service-Token) for multi-tenant applications
"""

# Base client and exceptions
from .client import (
    RecallBricksClient,
    RecallBricksError,
    ValidationError,
    RateLimitError,
    CircuitBreakerError,
    APIError,
    TimeoutError,
    DeduplicationError,
    AuthenticationError,
    NotFoundError,
    ServiceUnavailableError,
    RateLimitInfo,
    CircuitBreaker,
    RateLimiter,
    MetricsCollector,
    get_session,
)

# Memories API
from .memories import (
    MemoriesAPI,
    Memory,
    ExtractedMetadata,
    LearnResult,
    SearchResult,
    RecallResult,
    ContextResult,
)

# Relationships API
from .relationships import (
    RelationshipsAPI,
    Relationship,
    RelatedMemory,
    RelationshipGraph,
    RelationshipTypeStats,
)

# Learning API
from .learning import (
    LearningAPI,
    LearningAnalysisResult,
    RelationshipSuggestion,
    MaintenanceSuggestion,
    LearningStatus,
    LearningMetrics,
)

# Collaboration API
from .collaboration import (
    CollaborationAPI,
    Agent,
    AgentPerformance,
    Contribution,
    Conflict,
    SynthesisResult,
)

# Agents API
from .agents import (
    AgentsAPI,
    AgentProfile,
    AgentContext,
    ContextInjectionResult,
    IdentityValidation,
    MemorySummary,
)

# Usage API
from .usage import (
    UsageAPI,
    UsageStats,
    UsageHistoryEntry,
    UsageBreakdown,
)

# Monitoring API
from .monitoring import (
    MonitoringAPI,
    HealthCheck,
    SLAMetrics,
    AuditLogEntry,
    ComponentHealth,
    DashboardMetrics,
    SystemInsight,
)

# LangChain integrations (backward compatible)
from .memory import RecallBricksMemory
from .chat_history import RecallBricksChatMessageHistory
from .retriever import RecallBricksRetriever

__version__ = "1.3.0"

__all__ = [
    # LangChain Integrations
    "RecallBricksMemory",
    "RecallBricksChatMessageHistory",
    "RecallBricksRetriever",
    # Base Client
    "RecallBricksClient",
    # API Modules
    "MemoriesAPI",
    "RelationshipsAPI",
    "LearningAPI",
    "CollaborationAPI",
    "AgentsAPI",
    "UsageAPI",
    "MonitoringAPI",
    # Data Classes - Memories
    "Memory",
    "ExtractedMetadata",
    "LearnResult",
    "SearchResult",
    "RecallResult",
    "ContextResult",
    # Data Classes - Relationships
    "Relationship",
    "RelatedMemory",
    "RelationshipGraph",
    "RelationshipTypeStats",
    # Data Classes - Learning
    "LearningAnalysisResult",
    "RelationshipSuggestion",
    "MaintenanceSuggestion",
    "LearningStatus",
    "LearningMetrics",
    # Data Classes - Collaboration
    "Agent",
    "AgentPerformance",
    "Contribution",
    "Conflict",
    "SynthesisResult",
    # Data Classes - Agents
    "AgentProfile",
    "AgentContext",
    "ContextInjectionResult",
    "IdentityValidation",
    "MemorySummary",
    # Data Classes - Usage
    "UsageStats",
    "UsageHistoryEntry",
    "UsageBreakdown",
    # Data Classes - Monitoring
    "HealthCheck",
    "SLAMetrics",
    "AuditLogEntry",
    "ComponentHealth",
    "DashboardMetrics",
    "SystemInsight",
    # Exceptions
    "RecallBricksError",
    "ValidationError",
    "RateLimitError",
    "CircuitBreakerError",
    "APIError",
    "TimeoutError",
    "DeduplicationError",
    "AuthenticationError",
    "NotFoundError",
    "ServiceUnavailableError",
    # Utilities
    "RateLimitInfo",
    "CircuitBreaker",
    "RateLimiter",
    "MetricsCollector",
    "get_session",
]
