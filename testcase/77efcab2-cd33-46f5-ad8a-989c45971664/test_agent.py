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
from agent import LLMService

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = LLMService()
    return instance

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_throughput(agent_instance):
    """Test processing throughput with generous threshold."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Comparative analysis result"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 200

    # Patch the _get_llm_client to return a mock client with async chat.completions.create
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(return_value=mock_response)
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(agent_instance, "_get_llm_client", return_value=mock_client):
        start_time = time.time()
        for _ in range(10):
            result = await agent_instance.generate_response(
                system_prompt="System prompt",
                user_prompt="User prompt",
                context_chunks=["Chunk 1", "Chunk 2"]
            )
            assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"