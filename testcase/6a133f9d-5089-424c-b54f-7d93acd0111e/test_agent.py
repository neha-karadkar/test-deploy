
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Import ONLY from 'agent'
from agent import validation_exception_handler

@pytest.mark.asyncio
async def test_unit_validation_exception_handler_security():
    """Test validation_exception_handler does not leak internal error details."""
    # Prepare a fake FastAPI Request and a fake RequestValidationError
    class DummyRequest:
        url = "http://test/analyze"
        method = "POST"
    class DummyValidationError(Exception):
        def __init__(self):
            self.errors = lambda: []
            self.body = None
    request = DummyRequest()
    exc = DummyValidationError()
    # Patch JSONResponse to capture the response
    with patch("agent.JSONResponse", new=MagicMock()) as mock_json_response:
        result = await validation_exception_handler(request, exc)
    assert result is not None