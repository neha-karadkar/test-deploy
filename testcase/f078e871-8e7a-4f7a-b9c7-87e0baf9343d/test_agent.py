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
from agent import AgentOrchestrator

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    # Patch both Azure Search and OpenAI external dependencies
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()), \
         patch("azure.search.documents.SearchClient", new=MagicMock()), \
         patch("azure.core.credentials.AzureKeyCredential", new=MagicMock()):
        instance = AgentOrchestrator()
    return instance

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_analyze_endpoint_throughput(agent_instance):
    """Test /analyze endpoint (AgentOrchestrator.process_query) throughput under load."""
    # Patch chunk_retriever.retrieve_chunks and llm_service.generate_response
    mock_chunks = ["Chunk 1 about Earth", "Chunk 2 about Jupiter"]
    mock_llm_response = (
        "Physical Dimensions:\n"
        "- Earth: 7,917.5 miles (12,742 km) (Source: Earth.pdf)\n"
        "- Jupiter: 88,846 miles (142,984 km) (Source: Jupiter.pdf)\n"
        "Size Comparison:\n"
        "- Jupiter is about 11 times wider than Earth (Source: Jupiter.pdf)\n"
        "Orbital Distances:\n"
        "- Earth: 93 million miles (150 million km) from Sun (Source: Earth.pdf)\n"
        "- Jupiter: 484 million miles (778 million km) from Sun (Source: Jupiter.pdf)\n"
        "Spatial Relationship:\n"
        "- Jupiter's orbit is much farther from the Sun than Earth's (Source: Jupiter.pdf)"
    )
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=mock_chunks)), \
         patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value=mock_llm_response)), \
         patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value=mock_llm_response)):
        start_time = time.time()
        # Simulate 10 concurrent requests (sequentially, as true concurrency is not possible in this test context)
        for _ in range(10):
            result = await agent_instance.process_query()
            assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"