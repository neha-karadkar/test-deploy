
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Import ONLY from 'agent'
from agent import AgentOrchestrator

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    # Patch both openai.AsyncAzureOpenAI and azure.search.documents.SearchClient
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()), \
         patch("azure.search.documents.SearchClient", new=MagicMock()):
        instance = AgentOrchestrator()
    return instance

# ── Integration Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_basic_functionality(agent_instance):
    """Basic functionality test."""
    # Patch the chunk retriever and llm_service to avoid real network calls
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=["chunk1", "chunk2"])), \
         patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value="Earth is smaller than Jupiter. (Source: Earth.pdf)")), \
         patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value="Earth is smaller than Jupiter. (Source: Earth.pdf)")):
        result = await agent_instance.process_query()
    assert result is not None