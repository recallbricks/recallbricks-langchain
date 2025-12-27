"""RecallBricks Retriever for LangChain

Provides a LangChain retriever using RecallBricks organized recall
for RAG (Retrieval Augmented Generation) applications.
"""

from typing import List, Optional, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
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


class RecallBricksRetriever(BaseRetriever):
    """
    LangChain retriever using RecallBricks organized recall.

    Features:
    - Organized recall with category summaries for better context
    - Automatic metadata in document objects
    - Rate limiting and error handling
    - Thread-safe operations

    Example:
        from langchain_recallbricks import RecallBricksRetriever
        from langchain.chains import RetrievalQA

        retriever = RecallBricksRetriever(
            api_key="your-api-key",
            k=5,
            organized=True
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=retriever
        )

        answer = qa_chain.run("What is cognitive memory infrastructure?")
    """

    # Pydantic v2 model config
    model_config = {"arbitrary_types_allowed": True}

    # Pydantic fields for configuration
    api_key: str
    api_url: str = "https://api.recallbricks.com/api/v1"
    project_id: Optional[str] = None
    k: int = 4
    organized: bool = True
    enable_logging: bool = True
    rate_limit: int = 100
    rate_limit_period: int = 60

    # Internal fields (excluded from Pydantic model via PrivateAttr)
    _rate_limiter: Any = None
    _lock: Any = None

    def __init__(
        self,
        api_key: str,
        project_id: Optional[str] = None,
        k: int = 4,
        organized: bool = True,
        api_url: str = "https://api.recallbricks.com/api/v1",
        enable_logging: bool = True,
        rate_limit: int = 100,
        rate_limit_period: int = 60,
        **kwargs
    ):
        """
        Initialize RecallBricks retriever.

        Args:
            api_key: RecallBricks API key
            project_id: Optional project ID for multi-tenant applications
            k: Number of documents to retrieve (default: 4)
            organized: Use organized recall with categories (default: True)
            api_url: RecallBricks API base URL
            enable_logging: Enable detailed logging
            rate_limit: Maximum requests per period
            rate_limit_period: Period in seconds
        """
        if not api_key:
            raise ValidationError("api_key is required")
        if not api_url.startswith('https://'):
            raise ValidationError(f"api_url must use HTTPS. Got: {api_url}")
        if k <= 0 or k > 100:
            raise ValidationError("k must be between 1 and 100")

        super().__init__(
            api_key=api_key,
            project_id=project_id,
            k=k,
            organized=organized,
            api_url=api_url,
            enable_logging=enable_logging,
            rate_limit=rate_limit,
            rate_limit_period=rate_limit_period,
            **kwargs
        )

        # Initialize internal state after super().__init__
        object.__setattr__(self, '_rate_limiter', RateLimiter(rate=rate_limit, per=rate_limit_period))
        object.__setattr__(self, '_lock', threading.Lock())

        if enable_logging:
            logger.info(f"RecallBricksRetriever initialized with k={k}, organized={organized}")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve relevant documents using RecallBricks organized recall.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of Document objects with content and metadata
        """
        if not self._rate_limiter.allow():
            raise RateLimitError("Rate limit exceeded. Please slow down requests.")

        url = f"{self.api_url}/memories/recall"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "limit": self.k,
            "organized": self.organized
        }

        if self.project_id:
            payload["project_id"] = self.project_id

        try:
            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            memories = data.get("memories", [])
            documents = []

            for memory in memories:
                text = memory.get("text", "")
                memory_metadata = memory.get("metadata", {})
                score = memory.get("score", 0.0)

                # Build document metadata
                doc_metadata = {
                    "score": score,
                    "memory_id": memory.get("id", ""),
                    "tags": memory_metadata.get("tags", []),
                    "category": memory_metadata.get("category", "General"),
                    "importance": memory_metadata.get("importance", 0.5),
                    "entities": memory_metadata.get("entities", []),
                    "created_at": memory.get("created_at", ""),
                }

                documents.append(
                    Document(
                        page_content=text,
                        metadata=doc_metadata
                    )
                )

            if self.enable_logging:
                logger.debug(f"Retrieved {len(documents)} documents for query: {query[:50]}...")

            return documents

        except requests.HTTPError as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RecallBricksError(f"Failed to retrieve documents: {e}")

        except Exception as e:
            logger.error(f"Unexpected error retrieving documents: {e}")
            raise RecallBricksError(f"Unexpected error: {e}")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Async version of document retrieval.

        Currently uses synchronous client. Future versions may implement
        true async support.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of Document objects
        """
        # For now, delegate to sync version
        # TODO: Implement true async support with aiohttp
        return self._get_relevant_documents(query, run_manager=run_manager)

    def get_relevant_documents_with_categories(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Retrieve documents with full category summaries.

        This method returns both the documents and the category summaries
        from organized recall, useful for building more context-aware prompts.

        Args:
            query: Search query

        Returns:
            Dictionary with 'documents' and 'categories' keys

        Example:
            result = retriever.get_relevant_documents_with_categories("user preferences")
            docs = result["documents"]
            categories = result["categories"]
            # Use categories to provide overview to LLM
        """
        if not self._rate_limiter.allow():
            raise RateLimitError("Rate limit exceeded. Please slow down requests.")

        url = f"{self.api_url}/memories/recall"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "limit": self.k,
            "organized": True  # Force organized for this method
        }

        if self.project_id:
            payload["project_id"] = self.project_id

        try:
            response = get_session().post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            memories = data.get("memories", [])
            categories = data.get("categories", {})
            documents = []

            for memory in memories:
                text = memory.get("text", "")
                memory_metadata = memory.get("metadata", {})
                score = memory.get("score", 0.0)

                doc_metadata = {
                    "score": score,
                    "memory_id": memory.get("id", ""),
                    "tags": memory_metadata.get("tags", []),
                    "category": memory_metadata.get("category", "General"),
                    "importance": memory_metadata.get("importance", 0.5),
                }

                documents.append(
                    Document(
                        page_content=text,
                        metadata=doc_metadata
                    )
                )

            return {
                "documents": documents,
                "categories": categories
            }

        except requests.HTTPError as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RecallBricksError(f"Failed to retrieve documents: {e}")
