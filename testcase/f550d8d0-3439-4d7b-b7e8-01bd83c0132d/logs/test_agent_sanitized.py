
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Import ONLY from 'agent'
from agent import AgentOrchestrator, Config

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = AgentOrchestrator()
    return instance

# ── Security Tests ─────────────────────────────────────────────────

def test_security_config_validate_missing_api_key():
    """Validates that Config.validate() raises ValueError if required API keys are missing for Azure provider."""
    # Save original values to restore after test
    orig_provider = getattr(Config, "MODEL_PROVIDER", None) or None or None
    orig_api_key = getattr(Config, "AZURE_OPENAI_API_KEY", None) or None or None
    try:
        setattr(Config, "MODEL_PROVIDER", "azure")
        setattr(Config, "AZURE_OPENAI_API_KEY", "")
        with pytest.raises(ValueError) as exc_info:
            Config.validate()
        assert "AZURE_OPENAI_API_KEY is required" in str(exc_info.value)
    finally:
        # Restore original values
        setattr(Config, "MODEL_PROVIDER", orig_provider)
        setattr(Config, "AZURE_OPENAI_API_KEY", orig_api_key)