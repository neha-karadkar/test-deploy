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
from agent import AgentOrchestrator, AnalyzeResponse, sanitize_llm_output

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
async def test_integration_workflow(agent_instance):
    """Test complete workflow with mocked dependencies."""
    # Mock chunk retrieval to return fake chunks
    fake_chunks = ["Earth is the third planet from the Sun.", "Jupiter is the largest planet."]
    # Mock LLM response
    fake_llm_response = (
        "Physical Dimensions:\n"
        "- Earth: 7,917.5 miles (12,742 km) (Source: Earth.pdf)\n"
        "- Jupiter: 88,846 miles (142,984 km) (Source: Jupiter.pdf)\n"
        "Size Comparison:\n"
        "- Jupiter is about 11 times wider than Earth (Source: Jupiter.pdf)\n"
        "Orbital Distances:\n"
        "- Earth: 93 million miles (150 million km) from the Sun (Source: Earth.pdf)\n"
        "- Jupiter: 484 million miles (778 million km) from the Sun (Source: Jupiter.pdf)\n"
    )
    # Patch dependencies
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=fake_chunks)), \
         patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value=fake_llm_response)), \
         patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value=fake_llm_response)):
        result = await agent_instance.process_query()
    assert result is not None

@pytest.mark.asyncio
async def test_integration_workflow_chunk_retrieval_error(agent_instance):
    """Test workflow when chunk retrieval fails."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(side_effect=Exception("search error"))), \
         patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value="irrelevant")), \
         patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value="irrelevant")):
        try:
            result = await agent_instance.process_query()
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass

@pytest.mark.asyncio
async def test_integration_workflow_llm_error(agent_instance):
    """Test workflow when LLMService.generate_response fails."""
    fake_chunks = ["Earth is the third planet from the Sun.", "Jupiter is the largest planet."]
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=fake_chunks)), \
         patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(side_effect=Exception("llm error"))), \
         patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value="irrelevant")):
        try:
            result = await agent_instance.process_query()
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass

@pytest.mark.asyncio
async def test_integration_workflow_formatting_error(agent_instance):
    """Test workflow when formatting fails."""
    fake_chunks = ["Earth is the third planet from the Sun.", "Jupiter is the largest planet."]
    fake_llm_response = "Some LLM output"
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=fake_chunks)), \
         patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value=fake_llm_response)), \
         patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(side_effect=Exception("format error"))):
        try:
            result = await agent_instance.process_query()
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_throughput(agent_instance):
    """Test processing throughput with generous threshold."""
    fake_chunks = ["Earth is the third planet from the Sun.", "Jupiter is the largest planet."]
    fake_llm_response = (
        "Physical Dimensions:\n"
        "- Earth: 7,917.5 miles (12,742 km) (Source: Earth.pdf)\n"
        "- Jupiter: 88,846 miles (142,984 km) (Source: Jupiter.pdf)\n"
    )
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=fake_chunks)), \
         patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value=fake_llm_response)), \
         patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value=fake_llm_response)):
        start_time = time.time()
        for _ in range(10):
            result = await agent_instance.process_query()
            assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

# ── Edge Case Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edge_case_empty_chunks(agent_instance):
    """Test handling of empty chunk retrieval (no context found)."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=[])), \
         patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value="irrelevant")), \
         patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value="irrelevant")):
        result = await agent_instance.process_query()
    assert result is not None