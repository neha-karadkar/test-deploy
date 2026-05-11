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

# Import ONLY from 'agent' — the file is always agent.py
from agent import AgentOrchestrator

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = AgentOrchestrator()
    return instance

# ── Unit Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unit_process_query_happy_path(agent_instance):
    """Test process_query returns expected result."""
    mock_chunks = [
        "Earth's equatorial diameter is 7,926 miles (Source: Earth.pdf).",
        "Jupiter's equatorial diameter is 88,846 miles (Source: Jupiter.pdf)."
    ]
    mock_llm_response = (
        "Physical Dimensions:\n"
        "- Earth: 7,926 miles (Source: Earth.pdf)\n"
        "- Jupiter: 88,846 miles (Source: Jupiter.pdf)\n"
        "Size Comparison:\n"
        "- Jupiter is about 11 times wider than Earth.\n"
        "Orbital Distances:\n"
        "- Earth: 93 million miles from Sun (Source: Earth.pdf)\n"
        "- Jupiter: 484 million miles from Sun (Source: Jupiter.pdf)\n"
    )
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=mock_chunks)):
        with patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value=mock_llm_response)):
            with patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value=mock_llm_response)):
                result = await agent_instance.process_query()
    assert result is not None

@pytest.mark.asyncio
async def test_unit_process_query_error_handling(agent_instance):
    """Test process_query handles errors gracefully."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(side_effect=Exception("test error"))):
        try:
            result = await agent_instance.process_query()
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise  # ALWAYS reraise — do NOT swallow real test assertion failures
        except Exception:
            pass  # Agent propagated the error — also valid behavior

# ── Integration Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_integration_workflow(agent_instance):
    """Test complete workflow with mocked dependencies."""
    mock_chunks = [
        "Earth's equatorial diameter is 7,926 miles (Source: Earth.pdf).",
        "Jupiter's equatorial diameter is 88,846 miles (Source: Jupiter.pdf)."
    ]
    mock_llm_response = (
        "Physical Dimensions:\n"
        "- Earth: 7,926 miles (Source: Earth.pdf)\n"
        "- Jupiter: 88,846 miles (Source: Jupiter.pdf)\n"
        "Size Comparison:\n"
        "- Jupiter is about 11 times wider than Earth.\n"
        "Orbital Distances:\n"
        "- Earth: 93 million miles from Sun (Source: Earth.pdf)\n"
        "- Jupiter: 484 million miles from Sun (Source: Jupiter.pdf)\n"
    )
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=mock_chunks)):
        with patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value=mock_llm_response)):
            with patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value=mock_llm_response)):
                result = await agent_instance.process_query()
    assert result is not None

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_throughput(agent_instance):
    """Test processing throughput with generous threshold."""
    mock_chunks = [
        "Earth's equatorial diameter is 7,926 miles (Source: Earth.pdf).",
        "Jupiter's equatorial diameter is 88,846 miles (Source: Jupiter.pdf)."
    ]
    mock_llm_response = (
        "Physical Dimensions:\n"
        "- Earth: 7,926 miles (Source: Earth.pdf)\n"
        "- Jupiter: 88,846 miles (Source: Jupiter.pdf)\n"
        "Size Comparison:\n"
        "- Jupiter is about 11 times wider than Earth.\n"
        "Orbital Distances:\n"
        "- Earth: 93 million miles from Sun (Source: Earth.pdf)\n"
        "- Jupiter: 484 million miles from Sun (Source: Jupiter.pdf)\n"
    )
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=mock_chunks)):
        with patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value=mock_llm_response)):
            with patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value=mock_llm_response)):
                start_time = time.time()
                for _ in range(10):
                    result = await agent_instance.process_query()
                    assert result is not None
                duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

# ── Edge Case Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edge_case_empty_input(agent_instance):
    """Test handling of empty/None input."""
    empty_chunks = []
    empty_response = ""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=empty_chunks)):
        with patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(return_value=empty_response)):
            with patch.object(agent_instance.response_formatter, "format_response", new=MagicMock(return_value=empty_response)):
                result = await agent_instance.process_query()
    assert result is not None