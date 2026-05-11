# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")

# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Import ONLY from 'agent'
from agent import ChunkRetriever

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def chunk_retriever_instance():
    """Create ChunkRetriever with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = ChunkRetriever()
    return instance

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_retrieve_chunks_latency(chunk_retriever_instance):
    """Test embedding and search latency in ChunkRetriever.retrieve_chunks."""
    # Patch openai.AsyncAzureOpenAI.embeddings.create and SearchClient.search
    mock_embedding = MagicMock()
    mock_embedding.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    with patch("openai.AsyncAzureOpenAI") as mock_openai_client, \
         patch("azure.search.documents.SearchClient") as mock_search_client_class, \
         patch("azure.core.credentials.AzureKeyCredential", new=MagicMock()), \
         patch("azure.search.documents.models.VectorizedQuery", new=MagicMock()), \
         patch("agent.trace_tool_call", new=MagicMock()):
        # Setup mock openai client
        mock_openai_instance = MagicMock()
        mock_openai_instance.embeddings.create = AsyncMock(return_value=mock_embedding)
        mock_openai_client.return_value = mock_openai_instance

        # Setup mock search client
        mock_search_instance = MagicMock()
        mock_search_instance.search = MagicMock(return_value=[{"chunk": "Earth is a planet."}, {"chunk": "Jupiter is a gas giant."}])
        mock_search_client_class.return_value = mock_search_instance

        # Patch _get_search_client to use our mock
        chunk_retriever_instance.search_client = mock_search_instance

        start_time = time.time()
        result = await chunk_retriever_instance.retrieve_chunks(
            query="Compare Earth and Jupiter",
            filter_titles=["Earth.pdf", "Jupiter.pdf"],
            top_k=2
        )
        assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"retrieve_chunks took {duration:.1f}s"