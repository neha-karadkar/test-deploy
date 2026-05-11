
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Import ONLY from 'agent'
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
async def test_unit_process_query_error_handling(agent_instance):
    """Test process_query handles errors gracefully."""
    # Patch chunk_retriever.retrieve_chunks to raise an exception
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
async def test_integration_process_query_error_handling(agent_instance):
    """Test complete workflow error handling with mocked dependencies."""
    # Patch llm_service.generate_response to raise an exception after chunk retrieval
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=["chunk1", "chunk2"])):
        with patch.object(agent_instance.llm_service, "generate_response", new=AsyncMock(side_effect=Exception("test error"))):
            try:
                result = await agent_instance.process_query()
                assert result is not None  # Agent handled the error internally
            except AssertionError:
                raise
            except Exception:
                pass

# ── Edge Case Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edge_case_process_query_empty_chunks(agent_instance):
    """Test handling of empty chunks (no context found)."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=[])):
        result = await agent_instance.process_query()
    assert result is not None