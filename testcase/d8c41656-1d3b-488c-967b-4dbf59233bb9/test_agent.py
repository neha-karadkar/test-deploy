
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
import agent
from agent import app

@pytest.mark.asyncio
def test_basic_functionality():
    """
    Basic functionality test for the /analyze endpoint.
    Ensures that a POST to /analyze returns a non-None response.
    All external dependencies (Azure Search, OpenAI) are mocked.
    """
    # Patch ChunkRetriever.retrieve_chunks to return dummy chunks
    dummy_chunks = ["Earth is the third planet from the Sun.", "Jupiter is the largest planet."]
    # Patch LLMService.generate_response to return a dummy response
    dummy_llm_response = (
        "Physical Dimensions:\n"
        "- Earth: 7,917.5 miles (12,742 km) (Source: Earth.pdf)\n"
        "- Jupiter: 88,846 miles (142,984 km) (Source: Jupiter.pdf)\n"
        "Size Comparison:\n"
        "- Jupiter is about 11 times wider than Earth (Source: Jupiter.pdf)\n"
        "Orbital Distances:\n"
        "- Earth: 93 million miles (150 million km) from the Sun (Source: Earth.pdf)\n"
        "- Jupiter: 484 million miles (778 million km) from the Sun (Source: Jupiter.pdf)\n"
        "Spatial Relationship:\n"
        "- Jupiter's orbit is much farther from the Sun than Earth's (Source: Jupiter.pdf)"
    )

    # Patch the methods on the classes
    with patch.object(agent.ChunkRetriever, "retrieve_chunks", new_callable=AsyncMock) as mock_retrieve_chunks, \
         patch.object(agent.LLMService, "generate_response", new_callable=AsyncMock) as mock_generate_response:
        mock_retrieve_chunks.return_value = dummy_chunks
        mock_generate_response.return_value = dummy_llm_response

        # Use httpx.AsyncClient with ASGITransport for FastAPI app
        import httpx
        from httpx import AsyncClient, ASGITransport

        async def run_test():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/analyze")
                assert response is not None
                assert response.status_code == 200
                data = response.json()
                assert data is not None
                assert data.get("success") is True
                assert isinstance(data.get("result"), str)
                assert "Earth" in data.get("result")
                assert "Jupiter" in data.get("result")

        import asyncio
        asyncio.run(run_test())