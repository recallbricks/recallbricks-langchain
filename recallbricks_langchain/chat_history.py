"""RecallBricks Chat Message History for LangChain

Provides persistent chat message history backed by RecallBricks
with automatic metadata extraction.
"""

from typing import List, Optional, Dict, Any
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import requests
import logging
import threading

from recallbricks_langchain.client import (
    get_session,
    RecallBricksError,
    ValidationError,
    RateLimitError,
    CircuitBreaker,
    RateLimiter,
)

logger = logging.getLogger(__name__)


class RecallBricksChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history backed by RecallBricks with automatic metadata extraction.

    Features:
    - Persistent message storage across sessions
    - Automatic metadata extraction via learn() endpoint
    - Session-based message isolation
    - Thread-safe operations

    Example:
        from langchain_recallbricks import RecallBricksChatMessageHistory
        from langchain.memory import ConversationBufferMemory

        history = RecallBricksChatMessageHistory(
            api_key="your-api-key",
            session_id="user-123-session-1"
        )

        memory = ConversationBufferMemory(
            chat_memory=history,
            return_messages=True
        )
    """

    def __init__(
        self,
        api_key: str,
        session_id: str,
        project_id: Optional[str] = None,
        api_url: str = "https://api.recallbricks.com/api/v1",
        rate_limit: int = 100,
        rate_limit_period: int = 60,
        enable_logging: bool = True,
    ):
        """
        Initialize RecallBricks chat message history.

        Args:
            api_key: RecallBricks API key
            session_id: Unique session identifier for message isolation
            project_id: Optional project ID for multi-tenant applications
            api_url: RecallBricks API base URL
            rate_limit: Maximum requests per period
            rate_limit_period: Period in seconds
            enable_logging: Enable detailed logging
        """
        if not api_key:
            raise ValidationError("api_key is required")
        if not session_id:
            raise ValidationError("session_id is required")
        if not api_url.startswith('https://'):
            raise ValidationError(f"api_url must use HTTPS. Got: {api_url}")

        self.api_key = api_key
        self.session_id = session_id
        self.project_id = project_id
        self.api_url = api_url.rstrip('/')
        self.enable_logging = enable_logging

        # Rate limiter
        self.rate_limiter = RateLimiter(rate=rate_limit, per=rate_limit_period)

        # Local message cache
        self._messages: List[BaseMessage] = []
        self._loaded = False
        self._lock = threading.Lock()

        if enable_logging:
            logger.info(f"RecallBricksChatMessageHistory initialized for session: {session_id}")

    @property
    def messages(self) -> List[BaseMessage]:
        """
        Retrieve messages from RecallBricks.

        Loads messages from the API on first access, then returns cached messages.

        Returns:
            List of BaseMessage objects
        """
        with self._lock:
            if not self._loaded:
                self._load_messages()
            return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to RecallBricks with automatic metadata extraction.

        Args:
            message: Message to add (HumanMessage or AIMessage)
        """
        if not self.rate_limiter.allow():
            raise RateLimitError("Rate limit exceeded. Please slow down requests.")

        # Determine message type prefix
        if isinstance(message, HumanMessage):
            prefix = "human"
        elif isinstance(message, AIMessage):
            prefix = "ai"
        else:
            prefix = message.type

        # Save with automatic metadata extraction via learn()
        url = f"{self.api_url}/memories/learn"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": f"{prefix}: {message.content}",
            "source": f"langchain-session-{self.session_id}",
            "metadata": {
                "session_id": self.session_id,
                "message_type": prefix
            }
        }

        if self.project_id:
            payload["project_id"] = self.project_id

        try:
            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            # Add to local cache
            with self._lock:
                self._messages.append(message)

            if self.enable_logging:
                logger.debug(f"Added {prefix} message to session {self.session_id}")

        except requests.HTTPError as e:
            logger.error(f"Failed to add message: {e}")
            raise RecallBricksError(f"Failed to add message: {e}")

    def add_user_message(self, message: str) -> None:
        """
        Add a user message.

        Args:
            message: Message content
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """
        Add an AI message.

        Args:
            message: Message content
        """
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """
        Clear local message cache.

        Note: RecallBricks memories persist and cannot be bulk deleted.
        This only clears the local cache.
        """
        with self._lock:
            self._messages = []
            self._loaded = False

        if self.enable_logging:
            logger.info(f"Cleared local cache for session {self.session_id}")

    def _load_messages(self) -> None:
        """
        Load message history from RecallBricks.

        Queries memories for this session and reconstructs the message list.
        """
        url = f"{self.api_url}/memories/recall"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": f"session:{self.session_id}",
            "limit": 100,
            "organized": False
        }

        if self.project_id:
            payload["project_id"] = self.project_id

        try:
            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            memories = data.get("memories", [])

            # Convert memories back to messages
            for memory in memories:
                text = memory.get("text", "")
                metadata = memory.get("metadata", {})

                # Check metadata first for message type
                message_type = metadata.get("message_type", "")

                if message_type == "human" or text.lower().startswith("human:"):
                    content = text[6:].strip() if text.lower().startswith("human:") else text
                    self._messages.append(HumanMessage(content=content))
                elif message_type == "ai" or text.lower().startswith("ai:"):
                    content = text[3:].strip() if text.lower().startswith("ai:") else text
                    self._messages.append(AIMessage(content=content))

            self._loaded = True

            if self.enable_logging:
                logger.debug(f"Loaded {len(self._messages)} messages for session {self.session_id}")

        except requests.HTTPError as e:
            logger.error(f"Failed to load messages: {e}")
            self._messages = []
            self._loaded = True

        except Exception as e:
            logger.error(f"Unexpected error loading messages: {e}")
            self._messages = []
            self._loaded = True
